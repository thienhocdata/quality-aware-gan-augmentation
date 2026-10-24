"""Execute the generated master notebook in its CPU-only default mode."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "00_MASTER_CURRENT.ipynb"
TEMP = ROOT / "notebooks" / "00_MASTER_CURRENT.executing.ipynb"


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    executed = client.execute()
    nbformat.write(executed, TEMP)
    TEMP.replace(NOTEBOOK)
    print("00_MASTER_CURRENT.ipynb executed successfully")


if __name__ == "__main__":
    main()
