"""Local GPU experiment setup and a systemd cgroup with bounded RAM and swap.

Policy: RAM first, Spark spill on SSD next, OS swap only up to RETAIL_JOB_SWAP_MAX.
"""
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def load_env():
    path = PROJECT / '.env'
    for number, line in enumerate(path.read_text().splitlines() if path.exists() else [], 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.removeprefix('export ').partition('=')
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key.strip()):
            raise ValueError(f'Invalid .env line {number}')
        parts = shlex.split(value, comments=True)
        if len(parts) > 1:
            raise ValueError(f'Quote spaces in .env line {number}')
        os.environ.setdefault(key.strip(), parts[0] if parts else '')
    defaults = {
        'RETAIL_DATA_REVISION': '79bcf78cd4699267f561ad3fd51fe7b52af53083',
        'SPARK_MASTER': 'local[2]', 'SPARK_DRIVER_MEMORY': '3g',
        'SPARK_EXECUTOR_MEMORY': '3g', 'SPARK_SHUFFLE_PARTITIONS': '64',
        'RETAIL_JOB_MEMORY_MAX': '7G', 'RETAIL_JOB_SWAP_MAX': '4G', 'RETAIL_GPU_HOST_BUDGET_GB': '2.5',
        'RETAIL_GPU_USER_BATCH': '64', 'RETAIL_GPU_ITEM_BATCH': '8192',
        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '2',
        'MALLOC_ARENA_MAX': '2', 'MALLOC_TRIM_THRESHOLD_': '131072',
        'ARROW_DEFAULT_MEMORY_POOL': 'system',
        'SPARK_LOCAL_IP': '127.0.0.1',
        'SPARK_LOCAL_DIRS': str(PROJECT / '.spark-local'),
        'PYSPARK_PYTHON': sys.executable,
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    # These libraries must be on the loader path before starting the managed
    # Python process (changing LD_LIBRARY_PATH after dlopen is too late).
    site = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    library_paths = [str(site / 'nvidia/cu13/lib')]
    library_paths += os.environ.get('LD_LIBRARY_PATH', '').split(':')
    os.environ['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(p for p in library_paths if p))
    Path(os.environ['SPARK_LOCAL_DIRS']).mkdir(exist_ok=True, parents=True)
    homes = [Path(os.environ['JAVA_HOME'])] if os.getenv('JAVA_HOME') else [
        Path('/usr/lib/jvm/java-21-openjdk'), Path('/usr/lib/jvm/java-21-openjdk-amd64')]
    for home in homes:
        java = home / 'bin/java'
        if java.exists():
            result = subprocess.run([str(java), '-version'], capture_output=True, text=True)
            if re.search(r'version "21[.]', result.stderr + result.stdout):
                os.environ['JAVA_HOME'] = str(home)
                return
    raise RuntimeError('Set JAVA_HOME to JDK 21 before starting Spark.')


def memory_scope():
    relative = next(line.split('::', 1)[1].strip() for line in
                    Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    directory = Path('/sys/fs/cgroup') / relative.lstrip('/')
    return {name: (directory / name).read_text().strip() for name in
            ['memory.max', 'memory.swap.max', 'memory.current', 'memory.swap.current']}


def managed_entrypoint():
    """Re-exec the command and all its kernels/JVMs inside a bounded RAM/swap scope."""
    load_env()
    if os.getenv('RETAIL_MANAGED_SCOPE') == '1':
        state = memory_scope()
        if 'max' in (state['memory.max'], state['memory.swap.max']):
            raise RuntimeError('RAM/swap limits were not applied; refusing an unbounded run.')
        return
    if not shutil.which('systemd-run'):
        raise RuntimeError('systemd-run --user is required for the bounded RAM/swap launcher.')
    env = os.environ.copy()
    env['RETAIL_MANAGED_SCOPE'] = '1'
    command = ['systemd-run', '--user', '--scope', '--quiet',
               '-p', 'MemorySwapMax=' + env['RETAIL_JOB_SWAP_MAX'],
               '-p', 'MemoryMax=' + env['RETAIL_JOB_MEMORY_MAX'],
               sys.executable, '-u', *sys.argv]
    raise SystemExit(subprocess.call(command, env=env))
