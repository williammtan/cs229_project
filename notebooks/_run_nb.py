"""Generic streaming notebook executor: live stdout + incremental saves.

Usage: python _run_nb.py <notebook.ipynb>
Streams each cell's stream/error output to real stdout as it arrives and writes
the executed notebook back in place after every cell (crash-resilient).
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB_PATH = Path(sys.argv[1]).resolve()
_real_out = sys.__stdout__


def emit(text: str) -> None:
    _real_out.write(text)
    _real_out.flush()


class StreamingClient(NotebookClient):
    def output(self, outs, msg, display_id, cell_index):
        msg_type = msg["header"]["msg_type"]
        content = msg.get("content", {})
        if msg_type == "stream":
            emit(content.get("text", ""))
        elif msg_type == "error":
            emit("\n".join(content.get("traceback", [])) + "\n")
        return super().output(outs, msg, display_id, cell_index)


def main() -> int:
    nb = nbformat.read(NB_PATH, as_version=4)
    client = StreamingClient(nb, timeout=-1, kernel_name="etm_clf",
                             allow_errors=False, record_timing=True)
    code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    rc = 0
    with client.setup_kernel():
        for n, (i, cell) in enumerate(code_cells):
            emit(f"\n===== CELL {i} ({n + 1}/{len(code_cells)}) =====\n")
            try:
                client.execute_cell(cell, i)
            except CellExecutionError as e:
                emit(f"\n!!! CELL {i} FAILED: {e}\n")
                rc = 1
                nbformat.write(nb, NB_PATH)
                break
            nbformat.write(nb, NB_PATH)
            emit(f"----- cell {i} done -----\n")
    nbformat.write(nb, NB_PATH)
    emit(f"\n===== EXECUTION {'FAILED' if rc else 'COMPLETE'} =====\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
