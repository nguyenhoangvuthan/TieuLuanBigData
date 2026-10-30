"""Register the project's Python 3.14 GPU kernel after installing requirements-gpu.txt."""
import json
import sys
import tempfile
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager

project = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    (directory / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, str(project / 'tools/gpu_kernel.py'), '-f', '{connection_file}'],
        'display_name': 'Retail ALS GPU (CUDA / bounded swap)', 'language': 'python',
        'env': {'JAVA_HOME': '/usr/lib/jvm/java-21-openjdk'},
    }, indent=2))
    installed = KernelSpecManager().install_kernel_spec(str(directory),
        kernel_name='retail-als-gpu', user=True)
    print('GPU kernel installed:', installed)
