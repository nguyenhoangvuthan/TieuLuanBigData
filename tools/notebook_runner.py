"""Run the notebook top to bottom from a terminal, streaming its output.

Resume is implicit: the notebook keeps its own checkpoint
(results/<RUN_ID>/checkpoint/state.json) and skips completed stages, so a
rerun after a crash or power loss continues where the last commit stopped.
"""
import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT = Path(__file__).resolve().parents[1]
DISPLAY_LIMIT = 4000


class StreamingNotebookClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        kind, content = msg.get('msg_type'), msg.get('content', {})
        if kind == 'stream':
            print(content.get('text', ''), end='', flush=True)
        elif kind in ('display_data', 'execute_result'):
            text = content.get('data', {}).get('text/plain', '')
            if text and 'image/png' not in content.get('data', {}):
                print(text[:DISPLAY_LIMIT], flush=True)
        elif kind == 'error':
            print('\n'.join(content.get('traceback', [])), flush=True)
        return result


def write_atomic(nb, path):
    temporary = path.with_name(path.name + '.tmp')
    nbformat.write(nb, temporary)
    os.replace(temporary, path)


def run():
    os.chdir(PROJECT)
    os.environ['RETAIL_TERMINAL_RUN'] = '1'
    source_path = PROJECT / 'tieu_luan_retail_als.ipynb'
    source = os.getenv('RETAIL_SOURCE', 'full')
    run_id = os.getenv('RETAIL_STAR_RUN_ID') or f"retail_star_{source if source in ('full', 'small') else 'custom'}_v1"
    execution = PROJECT / 'results' / run_id / 'execution'
    execution.mkdir(parents=True, exist_ok=True)
    executed = execution / 'latest.ipynb'
    nb = nbformat.read(source_path, as_version=4)
    client = StreamingNotebookClient(nb, timeout=None, kernel_name=os.getenv('RETAIL_KERNEL', 'retail-als-gpu'),
        resources={'metadata': {'path': str(PROJECT)}}, record_timing=True)
    heading = ''
    with client.setup_kernel():
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                titles = [line for line in cell.source.splitlines() if line.startswith('#')]
                heading = titles[-1].lstrip('# ').strip() if titles else heading
                continue
            print(f'CELL_START {index} {time.strftime("%H:%M:%S")} | {heading}', flush=True)
            started = time.monotonic()
            try:
                client.execute_cell(cell, index)
            except Exception:
                write_atomic(nb, executed)
                print(f'CELL_FAILED {index} | {heading}\nEXECUTED_COPY {executed}', flush=True)
                raise
            write_atomic(nb, executed)
            print(f'CELL_PASSED {index} elapsed={time.monotonic() - started:.1f}s', flush=True)
    print(f'NOTEBOOK_ALL_CELLS_PASSED {executed}', flush=True)
