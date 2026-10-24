"""Rebuild the lightweight source/artifact registry for KLTN_clean.

This script never trains a model, loads TensorFlow, opens test tensors, or changes
historical artifacts. It hashes small source files, validates Python/notebook
syntax, and records the already-pinned identity of large archives.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "provenance" / "current"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(
    source_id: str,
    phase: str,
    path: str,
    role: str,
    evidence_status: str,
    expected_sha256: str,
    notes: str = "",
) -> dict[str, str]:
    return {
        "source_id": source_id,
        "phase": phase,
        "path": path,
        "role": role,
        "evidence_status": evidence_status,
        "expected_sha256": expected_sha256,
        "notes": notes,
    }


SOURCES = [
    source(
        "paper_notebook",
        "00_historical_reproduction",
        "archive/executed/3DHyperGAMO_IndianPines_Paper_Reproduction.ipynb",
        "Historical reproduction notebook; protocol is separate from Protocol B/M4.",
        "historical_source_preserved",
        "05349fb398b6aa0492a782d054cedbe023498b58e9da85c2f0c88aefa822097f",
    ),
    source("paper_author_auxil", "00_historical_reproduction", "src/frozen/paper_reproduction/author_auxil.py", "Historical reproduction dependency.", "historical_source_preserved", "f74bc64e5ae7d5b9fb5bab402eda6744cb39dfa1eece0c5478640ffd2783f79c"),
    source("paper_author_dense", "00_historical_reproduction", "src/frozen/paper_reproduction/author_dense_suppli.py", "Historical reproduction dependency.", "historical_source_preserved", "0bbeacbbbdcee0bdfe703c13ea13b6e0147f0cb7c952417b505b67aefc969c51"),
    source("paper_dense3d", "00_historical_reproduction", "src/frozen/paper_reproduction/dense_net3D_SA_original.py", "Historical reproduction network source.", "historical_source_preserved", "bfcac057ae7996d501050c08e5701bd7b973a06688a8e2df9650d7fdfd6f59d6"),
    source("paper_dip_data", "00_historical_reproduction", "src/frozen/paper_reproduction/dip_data.py", "Historical reproduction data source.", "historical_source_preserved", "1cd10b55aa50b927169bee4b425e00e7716f9e4fd196b7ce50a6449d39e620e9"),
    source("paper_worker", "00_historical_reproduction", "src/frozen/paper_reproduction/hypergamo_worker.py", "Worker recovered from the historical output.", "historical_executed_source", "ca6c3dc2d563588ad5a3a0138d13fe50d3a5339f5b600f6bea70298c60c37685"),
    source("paper_main", "00_historical_reproduction", "src/frozen/paper_reproduction/main_original.py", "Historical reproduction entry point.", "historical_source_preserved", "4770860b39524f5065501823f665b2ad9ad84cc651e44f9ba6c51b75d93f47c5"),
    source(
        "protocol_b_executed_v8",
        "01_protocol_b",
        "archive/executed/Protocol_B_executed_V8_355611502.ipynb",
        "Executed Protocol B snapshot with saved outputs; authoritative for the historical B run.",
        "authoritative_executed_notebook",
        "a4b3e9ca6f657a683cd69fe957cbeaa527ab5bba64d7f3d53a803c514ef9e596",
    ),
    source(
        "protocol_b_source_reference",
        "01_protocol_b",
        "archive/legacy/KLTN_3D_HyperGAMO_Protocol_B_v2.ipynb",
        "Protocol B source reference; the executed V8 snapshot takes precedence for the actual run.",
        "authoritative_reference",
        "9424f236de4c2dbbef10d6015be068bf38dd75ce686ae9e8b851e38ab3aac467",
    ),
    source(
        "spatial_leakage_review",
        "01b_spatial_leakage_audit",
        "archive/legacy/KLTN_KiemChung_RoRiKhongGian_ReviewNhom.ipynb",
        "Spatial-overlap/leakage audit notebook; supporting limitation evidence, not the main training implementation.",
        "supporting_audit_source",
        "1e61b558611332ac58a90de846dff9cc345c0aca76c0c51cf1c8dc241593a7d8",
    ),
    source("v2_integration_smoke", "02_v2_technical", "src/frozen/smoke.py", "Exact integration-smoke worker.", "executed_source_pass_report", "dfdb2c59d4905c1504dc090eacd93965fc077d0543c3cd319d2923472d5713b7"),
    source("v2_matched_runner", "02_v2_technical", "src/frozen/runner.py", "Matched-runner technical check source.", "executed_source_pass_report", "3ac45bc3d22905904cd66a1f66ec03839192fb48a1af01f55528506dd4cd7992"),
    source("v2_final_technical", "02_v2_technical", "src/frozen/technical.py", "Final technical-check source; standalone report name was not recovered.", "source_preserved_report_unresolved", "1024fd19899d0ebecc8eae76b721333e3c63f2d8e5b0c04d4bf151ceb5ccb116"),
    source(
        "m4_training_worker",
        "03_m4_v2_training",
        "src/frozen/m4.py",
        "Authoritative worker for four arms x three seeds, including V2 Learned-local.",
        "authoritative_executed_source",
        "6d33ca6a0d39600f5f44c0d7e63547b1237149d625471ac73315d982ffbd8fb5",
    ),
    source(
        "m4_test_worker",
        "04_m4_saved_test",
        "src/frozen/test.py",
        "Authoritative executed 12-classifier test worker.",
        "authoritative_executed_source",
        "53673fd9ce094121431164053cc6cf990c7f4f4800aa599fc8ed88b86f88e4b9",
    ),
    source("v2_weighting_8000", "05_selector_diagnostics", "src/frozen/weighting.py", "Inference-only final-checkpoint learned-weight analysis.", "executed_analysis_source_complete", "131304f6155b56b04be601d888755445dafd332839d0d6d14c68a102fead872f"),
    source("v2_endgame_7500_8000", "05_selector_diagnostics", "src/frozen/endgame.py", "Corrected 7,500-to-8,000 diagnostic source.", "source_and_checkpoints_present_no_complete_report", "5e171cb7712c2569c99bfe957b375756e95c26d26243ed9cf8dccbd4f3a1a9a4"),
    source(
        "legacy_endpoint_named_file",
        "05_selector_diagnostics",
        "src/frozen/endpoint.py",
        "Older endpoint/endgame diagnostic source. It is not the six-job endpoint-ablation runner.",
        "preserved_but_not_final_endpoint_runner",
        "7c92c021459b825271aedee161194cde374b0867bb442421f645fd62e9d8048e",
        "Kept for provenance; do not use it to reproduce the final endpoint ablation.",
    ),
    source(
        "selector_endpoint_ablation_runner",
        "06_selector_endpoint_ablation",
        "src/frozen/selector_endpoint_ablation/endpoint_ablation_runner_clean_r3.py",
        "Exact runner extracted from the completed six-job endpoint-ablation backup.",
        "authoritative_executed_source",
        "5d1e0adf424ddad1467e1a2b420fe90861a6eaa2b7d3c74844e6de703cec64bf",
    ),
    source(
        "v3a_worker",
        "07_v3a_negative_ablation",
        "src/frozen/v3a/v3a_worker.py",
        "Exact V3-A distance-bias worker from the completed one-run backup.",
        "authoritative_executed_source",
        "d1d2105a63153f0c1e5859a31ca81611d65e4a1feee551e1bbfa299d04d315f7",
    ),
    source(
        "v3a_orchestrator",
        "07_v3a_negative_ablation",
        "src/frozen/v3a/v3a_orchestrator.py",
        "Exact V3-A audit/train/test orchestrator.",
        "authoritative_executed_source",
        "6a648c1024369d54ba3c357a85a2672d88a56f747d546c347d93973282cf71db",
    ),
    source(
        "v3a_pinned_m4_dependency",
        "07_v3a_negative_ablation",
        "src/frozen/v3a/m4_training_worker_frozen.py",
        "M4 source pinned inside the V3-A payload; must equal the authoritative M4 worker.",
        "authoritative_dependency_copy",
        "6d33ca6a0d39600f5f44c0d7e63547b1237149d625471ac73315d982ffbd8fb5",
    ),
]


ARTIFACTS = [
    {
        "artifact_id": "master_notebook_current",
        "phase": "08_current_project_review",
        "path": "notebooks/00_MASTER_CURRENT.ipynb",
        "status": "present_cpu_validated_15_cells",
        "bytes": 1101292,
        "sha256": "eba3b5cf3eef96c1db11233c1867c0c0dd06a3ec8128f7d8be6ee388cc743297",
        "role": "Current unified read-only notebook with embedded validated outputs through V3-A.",
    },
    {
        "artifact_id": "m4_classifiers_for_test",
        "phase": "04_m4_saved_test",
        "path": "checkpoints/M4_IP_12_CLASSIFIERS_FOR_TEST.zip",
        "status": "present_verified_reusable_bundle",
        "bytes": 37245118,
        "sha256": "295aebeae5ad067e7103d0f3c4f9efa1c3e31f4757cf9ec6c232ced46c490789",
        "role": "Compact bundle of all 12 classifiers for optional inference verification.",
    },
    {
        "artifact_id": "protocol_b_source_cache",
        "phase": "01_protocol_b",
        "path": "data/01_PROTOCOL_B_SOURCE.zip",
        "status": "present_hash_pinned",
        "bytes": 1408748991,
        "sha256": "8e586d26e65ecc4ff9567cc864509af7786cb1e32031cca4aae44cad331fd4fe",
        "role": "Protocol B data/cache/checkpoint source package.",
    },
    {
        "artifact_id": "m4_models_12",
        "phase": "03_m4_v2_training",
        "path": "checkpoints/02_M4_MODELS_12.zip",
        "status": "present_hash_pinned",
        "bytes": 1143820236,
        "sha256": "1c51b83de5bb80e777c9a730ab645e244e9b2a79edfbbc6b7588bd42c8dc15bf",
        "role": "Raw M4 12-job model/checkpoint archive.",
    },
    {
        "artifact_id": "learned_local_endgame_6_checkpoints",
        "phase": "05_selector_diagnostics",
        "path": "checkpoints/03_ENDGAME_LEARNED_LOCAL_6_CKPTS.zip",
        "status": "present_hash_pinned_analysis_pending",
        "bytes": 553574151,
        "sha256": "f14ecf08f2fe6e9efa73608d23e298660fde1a33c8285fd6e8d24e3ddd931903",
        "role": "Learned-local checkpoints at 7,500/8,000 for three seeds.",
    },
    {
        "artifact_id": "protocol_b_results",
        "phase": "01_protocol_b",
        "path": "outputs/bundles/01_PROTOCOL_B_RESULTS.zip",
        "status": "present_verified",
        "bytes": 3887245,
        "sha256": "92695c16366496cdd5fc0dc96876d4076508817e74c7e34cf2dfc01783660628",
        "role": "Lightweight Protocol B result bundle.",
    },
    {
        "artifact_id": "m4_training_results",
        "phase": "03_m4_v2_training",
        "path": "outputs/bundles/02_M4_TRAINING_RESULTS.zip",
        "status": "present_verified",
        "bytes": 395221,
        "sha256": "c9ec7de211a64afc4b4f8af0d861178adeee5f514e075bd071c2e38395112757",
        "role": "M4 training registry and reports.",
    },
    {
        "artifact_id": "m4_test_results",
        "phase": "04_m4_saved_test",
        "path": "outputs/bundles/03_M4_TEST_RESULTS.zip",
        "status": "present_verified",
        "bytes": 3034663,
        "sha256": "542cf047ff0e09b7f238a12eab483afac5badb3c3313d5b1d4e9b40c045941c7",
        "role": "Saved 12-model test predictions/metrics/confusions.",
    },
    {
        "artifact_id": "v2_classification_diagnostics",
        "phase": "05_selector_diagnostics",
        "path": "outputs/bundles/04_V2_CLASSIFICATION_DIAGNOSTICS.zip",
        "status": "present_verified",
        "bytes": 256880,
        "sha256": "28a7051a6cb786fd48ed3e842d9c1921cf2a3bed0fb70dd974734a495113371e",
        "role": "V2 classification diagnostics.",
    },
    {
        "artifact_id": "v2_weighting_8000",
        "phase": "05_selector_diagnostics",
        "path": "outputs/bundles/05_V2_WEIGHTING_8000.zip",
        "status": "present_verified_complete",
        "bytes": 16239039,
        "sha256": "4b390c7294bdbad32139a9ffbf27c824443556334d55494efe0e91b9d17eecad",
        "role": "Three-seed endpoint learned-weight analysis.",
    },
    {
        "artifact_id": "endgame_status",
        "phase": "05_selector_diagnostics",
        "path": "outputs/bundles/06_ENDGAME_7500_8000_STATUS.zip",
        "status": "present_failed_or_incomplete_status_only",
        "bytes": 4806,
        "sha256": "2fb38d19b47552007dcedc31dc9d183c255060def9f3b1ebee5df4f44c3c6fdc",
        "role": "Error/status evidence; not a completed 7,500-to-8,000 result.",
    },
    {
        "artifact_id": "selector_endpoint_analysis",
        "phase": "06_selector_endpoint_ablation",
        "path": "outputs/bundles/07_SELECTOR_ENDPOINT_ABLATION_ANALYSIS_20261019.zip",
        "status": "present_verified_complete",
        "bytes": 1497805,
        "sha256": "7fc8bb5145100d7e62326492c7e0d31dd58769fdb4e8a56b6063dd321668f96b",
        "role": "Lightweight audited analysis of all six endpoint-ablation jobs.",
    },
    {
        "artifact_id": "v3a_analysis",
        "phase": "07_v3a_negative_ablation",
        "path": "outputs/bundles/08_V3A_RESULTS_ANALYSIS_20261019.zip",
        "status": "present_complete_negative_ablation",
        "bytes": 1450676,
        "sha256": "5e0649b273d3a7af6fbb623a3a06496de04907fc5c6a829682b425c853aa9b09",
        "role": "Lightweight audited V3-A results and comparison with V2.",
    },
]


def validate_source(record: dict[str, str]) -> dict[str, str | int | bool]:
    path = ROOT / record["path"]
    result: dict[str, str | int | bool] = dict(record)
    result["exists"] = path.is_file()
    result["bytes"] = path.stat().st_size if path.is_file() else 0
    result["actual_sha256"] = sha256(path) if path.is_file() else ""
    result["hash_match"] = bool(path.is_file() and result["actual_sha256"] == record["expected_sha256"])
    syntax = "not_applicable"
    if path.is_file() and path.suffix == ".py":
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            syntax = "PASS"
        except Exception as exc:  # pragma: no cover - audit reporting
            syntax = f"FAIL: {type(exc).__name__}: {exc}"
    elif path.is_file() and path.suffix == ".ipynb":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            syntax = "PASS" if isinstance(payload.get("cells"), list) else "FAIL: no cells list"
        except Exception as exc:  # pragma: no cover - audit reporting
            syntax = f"FAIL: {type(exc).__name__}: {exc}"
    result["syntax_or_notebook_json"] = syntax
    return result


def validate_artifact(record: dict[str, str | int]) -> dict[str, str | int | bool]:
    path = ROOT / str(record["path"])
    result = dict(record)
    result["exists"] = path.is_file()
    result["actual_bytes"] = path.stat().st_size if path.is_file() else 0
    result["size_match"] = bool(path.is_file() and result["actual_bytes"] == record["bytes"])
    # Large archives retain hashes from the prior byte-level packaging audit.
    # Small bundles are re-hashed on every registry rebuild.
    if path.is_file() and path.stat().st_size <= 64 * 1024 * 1024:
        result["actual_sha256"] = sha256(path)
        expected = str(record.get("sha256", ""))
        result["hash_match"] = (not expected) or result["actual_sha256"] == expected
    else:
        result["actual_sha256"] = "not_rehashed_large_archive"
        result["hash_match"] = "pinned_from_prior_audit"
    return result


def write_csv(path: Path, records: list[dict]) -> None:
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [validate_source(record) for record in SOURCES]
    artifacts = [validate_artifact(record) for record in ARTIFACTS]
    source_ok = all(bool(row["exists"]) and bool(row["hash_match"]) and not str(row["syntax_or_notebook_json"]).startswith("FAIL") for row in sources)
    artifact_ok = all(bool(row["exists"]) and bool(row["size_match"]) and row["hash_match"] is not False for row in artifacts)
    cross_checks = {
        "m4_worker_equals_v3a_pinned_copy": (
            next(row for row in sources if row["source_id"] == "m4_training_worker")["actual_sha256"]
            == next(row for row in sources if row["source_id"] == "v3a_pinned_m4_dependency")["actual_sha256"]
        ),
        "legacy_endpoint_is_distinct_from_final_ablation_runner": (
            next(row for row in sources if row["source_id"] == "legacy_endpoint_named_file")["actual_sha256"]
            != next(row for row in sources if row["source_id"] == "selector_endpoint_ablation_runner")["actual_sha256"]
        ),
    }
    summary = {
        "schema": 2,
        "generated_date": date.today().isoformat(),
        "project": "KLTN_clean",
        "protocol_B_id": "50e988fae260",
        "scope": "Source and artifact identity audit only; no TensorFlow/model execution.",
        "source_records": len(sources),
        "artifact_records": len(artifacts),
        "source_identity_checks_pass": source_ok,
        "artifact_presence_checks_pass": artifact_ok,
        "cross_checks": cross_checks,
        "known_nonblocking_gaps": [
            "The 7,500-to-8,000 endgame forward analysis has no COMPLETE report.",
            "The standalone M3/final-technical report name remains unresolved, although its source is preserved.",
            "GPU package versions are recorded per historical run but are not a single bitwise environment lock.",
            "Protocol B is preserved as an executed notebook, not yet refactored into canonical reusable modules.",
        ],
        "safe_reorganization": "Historical bytes remain unchanged; missing executed sources were copied into src/frozen and 29 top-level legacy items were moved reversibly into indexed archives.",
    }
    (OUT / "source_registry.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "artifact_registry.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "registry_validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUT / "source_registry.csv", sources)
    write_csv(OUT / "artifact_registry.csv", artifacts)
    (ROOT / "CURRENT_PROJECT_MANIFEST.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
