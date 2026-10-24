"""Validate every code cell without launching a Jupyter kernel.

Used as a fallback on the local Windows host where jupyter.exe cannot start due
to an external DLL relocation error. This preserves notebook semantics closely
enough for the CPU-only saved-results path: one shared namespace, ordered cells,
and the same working directory. It does not synthesize notebook display outputs.
"""

from __future__ import annotations

import contextlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "00_MASTER_CURRENT.ipynb"
REPORT = ROOT / "docs" / "provenance" / "current" / "master_current_inprocess_validation.json"
LOG = ROOT / "docs" / "provenance" / "current" / "master_current_inprocess_validation.log"


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.chdir(ROOT)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "__main__"}
    records = []
    error = None
    with LOG.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            try:
                exec(compile(cell.source, f"{NOTEBOOK.name}:cell_{index}", "exec"), namespace)
                records.append({"cell": index, "status": "PASS"})
            except Exception as exc:  # pragma: no cover - validation report
                error = f"{type(exc).__name__}: {exc}"
                records.append({"cell": index, "status": "FAIL", "error": error})
                traceback.print_exc()
                break
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notebook": str(NOTEBOOK),
        "mode": "ordered in-process CPU validation; no Jupyter kernel",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_imported": "tensorflow" in __import__("sys").modules,
        "cells": records,
        "status": "PASS" if error is None else "FAIL",
        "error": error,
        "output_root": str(namespace.get("OUTPUT_ROOT", "")),
        "endpoint_status": namespace.get("ENDPOINT_STATUS"),
        "v3a_status": namespace.get("V3A_STATUS"),
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
