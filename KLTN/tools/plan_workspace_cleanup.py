"""Create a conservative cleanup plan for the KLTN workspace.

Only top-level loose files are checked for byte-identical copies in KLTN_clean.
Directories are classified by project role. This script does not move or delete.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
OUT = PROJECT / "docs" / "provenance" / "current"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_files_by_size() -> dict[int, list[Path]]:
    index: dict[int, list[Path]] = {}
    for path in PROJECT.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            index.setdefault(path.stat().st_size, []).append(path)
    return index


DIRECTORY_ROLES = {
    "KLTN_clean": ("KEEP_CANONICAL", "Project canonical."),
    "KLTN_archive_legacy": ("KEEP_LEGACY_ARCHIVE", "Reversible archive created after source/hash audit."),
    "outputs": ("KEEP_ARTIFACT_WAREHOUSE", "Large checkpoint/output warehouse."),
    "protocol_b_review": ("KEEP_AUDIT_ARCHIVE", "Research/audit history and received evidence."),
    "quality-aware-gan-augmentation": ("KEEP_SEPARATE_REPOSITORY", "Separate GitHub/release repository."),
    "KLTN_TRUC_CHINH": ("ARCHIVE_LEGACY_SNAPSHOT", "Older organized handover snapshot; canonical source is now in KLTN_clean."),
    "3DHyperGAMO_16class": ("ARCHIVE_HISTORICAL_BRANCH", "Historical experiment branch."),
    "3DHyperGAMO_paper": ("ARCHIVE_HISTORICAL_BRANCH", "Historical paper-reproduction workspace."),
    "3DHyperGAMO_spatial": ("ARCHIVE_HISTORICAL_BRANCH", "Historical spatial branch."),
    "four_method_outputs": ("ARCHIVE_HISTORICAL_OUTPUT", "Pre-Protocol-B/four-method outputs."),
    "split_audit_cells": ("ARCHIVE_HISTORICAL_BRANCH", "Split-audit construction files."),
    "qa_report_20260921": ("ARCHIVE_HISTORICAL_REPORT", "Old QA report."),
    "qa_report_20260921b": ("ARCHIVE_HISTORICAL_REPORT", "Old QA report."),
    "report_review": ("ARCHIVE_HISTORICAL_REPORT", "Old report review."),
    "__pycache__": ("DELETE_REGENERABLE_CACHE", "Python bytecode cache."),
    ".codex_tmp": ("DELETE_REGENERABLE_CACHE", "Temporary build files."),
    ".protocol_b_review_env": ("REVIEW_ENV_BEFORE_DELETE", "Local review environment; regenerable but may be large."),
    "node_modules": ("KEEP_SHARED_DEPENDENCY", "Workspace dependency installation; outside scientific evidence."),
    "_lit": ("KEEP_TEMPLATE_SUPPORT", "Local document/template support."),
    "_review_template": ("KEEP_TEMPLATE_SUPPORT", "Local review template support."),
    "docx_review_template": ("KEEP_TEMPLATE_SUPPORT", "Local DOCX review template support."),
    ".git": ("KEEP_WORKSPACE_METADATA", "Workspace Git metadata."),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = canonical_files_by_size()
    file_records: list[dict] = []
    for path in sorted((p for p in WORKSPACE.iterdir() if p.is_file()), key=lambda p: p.name.lower()):
        actual = sha256(path)
        matches = []
        for candidate in canonical.get(path.stat().st_size, []):
            if sha256(candidate) == actual:
                matches.append(candidate.relative_to(WORKSPACE).as_posix())
        classification = "REMOVE_EXACT_DUPLICATE_CANDIDATE" if matches else "KEEP_OR_MANUAL_REVIEW"
        file_records.append(
            {
                "path": path.relative_to(WORKSPACE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": actual,
                "classification": classification,
                "canonical_matches": ";".join(matches),
            }
        )

    directory_records = []
    for path in sorted((p for p in WORKSPACE.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        classification, note = DIRECTORY_ROLES.get(path.name, ("MANUAL_REVIEW", "Not classified automatically."))
        directory_records.append(
            {
                "path": path.name,
                "classification": classification,
                "note": note,
            }
        )

    payload = {
        "scope": "Conservative workspace cleanup plan; no move/delete performed.",
        "workspace": str(WORKSPACE),
        "canonical_project": str(PROJECT),
        "top_level_files": file_records,
        "top_level_directories": directory_records,
        "counts": {
            "exact_duplicate_file_candidates": sum(r["classification"] == "REMOVE_EXACT_DUPLICATE_CANDIDATE" for r in file_records),
            "manual_file_review": sum(r["classification"] != "REMOVE_EXACT_DUPLICATE_CANDIDATE" for r in file_records),
            "directories": len(directory_records),
        },
    }
    (OUT / "workspace_cleanup_plan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "workspace_cleanup_files.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(file_records[0]) if file_records else ["path"])
        writer.writeheader()
        writer.writerows(file_records)
    with (OUT / "workspace_cleanup_directories.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(directory_records[0]) if directory_records else ["path"])
        writer.writeheader()
        writer.writerows(directory_records)
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
