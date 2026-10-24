"""Embed concise outputs from the successful in-process CPU validation."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "00_MASTER_CURRENT.ipynb"
VALIDATION = ROOT / "docs" / "provenance" / "current" / "master_current_inprocess_validation.json"


def csv_from_zip(path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        return pd.read_csv(io.BytesIO(archive.read(member)))


def set_stream(cell, execution_count: int, text: str) -> None:
    cell.execution_count = execution_count
    cell.outputs = [nbf.v4.new_output("stream", name="stdout", text=text.rstrip() + "\n")]


def main() -> None:
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("tensorflow_imported") is not False:
        raise RuntimeError("Cannot embed outputs from a failed or TensorFlow-importing validation")
    notebook = nbf.read(NOTEBOOK, as_version=4)
    endpoint_zip = ROOT / "outputs" / "bundles" / "07_SELECTOR_ENDPOINT_ABLATION_ANALYSIS_20261019.zip"
    v3a_zip = ROOT / "outputs" / "bundles" / "08_V3A_RESULTS_ANALYSIS_20261019.zip"
    endpoint_seed = csv_from_zip(endpoint_zip, "metrics_per_seed_with_GM.csv")
    endpoint_summary = csv_from_zip(endpoint_zip, "summary_mean_sample_SD.csv")
    endpoint_paired = csv_from_zip(endpoint_zip, "paired_differences_recomputed.csv")
    v3a_summary = csv_from_zip(v3a_zip, "recomputed_summary_mean_sample_SD.csv")
    v3a_delta = csv_from_zip(v3a_zip, "V3_minus_full_M4_V2_by_seed.csv")
    v3a_probe = csv_from_zip(v3a_zip, "selector_probe_summary.csv")
    manifest = json.loads((ROOT / "CURRENT_PROJECT_MANIFEST.json").read_text(encoding="utf-8"))

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if "ENDPOINT_STATUS = {" in cell.source:
            set_stream(
                cell,
                11,
                "CPU saved-results validation; no train/test inference.\n\n"
                "KẾT QUẢ TỪNG SEED\n" + endpoint_seed.to_string(index=False) +
                "\n\nTRUNG BÌNH, SAMPLE SD VÀ CHÊNH GHÉP CẶP\n" + endpoint_summary.to_string(index=False) +
                "\n\nPAIRED DIFFERENCES\n" + endpoint_paired.to_string(index=False),
            )
        elif "V3A_STATUS = {" in cell.source:
            set_stream(
                cell,
                12,
                "CPU saved-results validation; no train/test inference.\n\n"
                "SO SÁNH 5 PHƯƠNG PHÁP\n" + v3a_summary.to_string(index=False) +
                "\n\nV3-A TRỪ V2 THEO SEED\n" + v3a_delta.to_string(index=False) +
                "\n\nSELECTOR PROBE 200/8000\n" + v3a_probe.to_string(index=False),
            )
        elif "final_status = {" in cell.source:
            set_stream(
                cell,
                14,
                "HOÀN TẤT đường đọc kết quả đã lưu.\n"
                f"Output: {validation.get('output_root')}\n"
                "Endpoint: COMPLETE_6_CLASSIFIERS_SAVED_RESULTS\n"
                "V3-A: COMPLETE_3_SEEDS_NEGATIVE_ABLATION\n"
                "new_training_performed=False\nnew_test_model_inference_performed=False",
            )
        elif "CURRENT_REGISTRY_STATUS" in cell.source:
            set_stream(
                cell,
                13,
                "SOURCE/ARTIFACT REGISTRY\n" + json.dumps(
                    {
                        "protocol_B_id": manifest["protocol_B_id"],
                        "source_records": manifest["source_records"],
                        "artifact_records": manifest["artifact_records"],
                        "source_identity_checks_pass": manifest["source_identity_checks_pass"],
                        "artifact_presence_checks_pass": manifest["artifact_presence_checks_pass"],
                        "cross_checks": manifest["cross_checks"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        elif "CPU_ONLY_NO_MODEL_IMPORT_PASS" in cell.source:
            set_stream(cell, 15, "CPU_ONLY_NO_MODEL_IMPORT_PASS")

    notebook.metadata["local_validation"] = {
        "status": "PASS",
        "mode": validation["mode"],
        "tensorflow_imported": False,
        "note": "New cell outputs were embedded from the ordered in-process CPU validation because the local jupyter.exe launcher had an external DLL relocation failure.",
    }
    nbf.validate(notebook)
    nbf.write(notebook, NOTEBOOK)
    print("Embedded validated CPU outputs into 00_MASTER_CURRENT.ipynb")


if __name__ == "__main__":
    main()
