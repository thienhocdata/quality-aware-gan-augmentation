"""Package the current read-only master review and its provenance."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "docs" / "provenance" / "current" / "master_current_inprocess_validation.json"
TARGET = ROOT / "outputs" / "bundles" / "09_MASTER_CURRENT_REVIEW_20261023.zip"
VERIFY = ROOT / "outputs" / "bundles" / "09_MASTER_CURRENT_REVIEW_20261023_VERIFICATION.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("Master validation is not PASS")
    review_root = Path(validation["output_root"])
    if not review_root.is_dir():
        raise RuntimeError(f"Missing review directory: {review_root}")
    explicit = [
        ROOT / "notebooks" / "00_MASTER_CURRENT.ipynb",
        ROOT / "CURRENT_PROJECT_MANIFEST.json",
        ROOT / "docs" / "SOURCE_COMPLETENESS_AUDIT_2026-10-23.md",
        ROOT / "docs" / "PROJECT_STRUCTURE_2026-10-23.md",
        ROOT / "docs" / "PHAN_6_KET_QUA_SELECTOR.md",
        VALIDATION,
        VALIDATION.with_suffix(".log"),
        ROOT / "docs" / "provenance" / "current" / "source_registry.json",
        ROOT / "docs" / "provenance" / "current" / "artifact_registry.json",
        ROOT / "docs" / "provenance" / "current" / "registry_validation.json",
    ]
    files: list[tuple[Path, str]] = []
    for path in explicit:
        if not path.is_file():
            raise RuntimeError(f"Missing package input: {path}")
        files.append((path, f"project/{path.relative_to(ROOT).as_posix()}"))
    for path in sorted(review_root.rglob("*")):
        if path.is_file():
            files.append((path, f"review/{path.relative_to(review_root).as_posix()}"))
    records = [
        {"member": member, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path, member in files
    ]
    package_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "CPU-only saved-results master review; no training or model inference",
        "protocol_B_id": "50e988fae260",
        "validation_status": validation["status"],
        "tensorflow_imported": validation["tensorflow_imported"],
        "members": records,
    }
    with zipfile.ZipFile(TARGET, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, member in files:
            archive.write(path, member)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(package_manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(TARGET) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
    verification = {
        "path": str(TARGET),
        "bytes": TARGET.stat().st_size,
        "sha256": sha256(TARGET),
        "crc_test": "PASS",
        "members": len(files) + 1,
        "training_performed": False,
        "model_inference_performed": False,
    }
    VERIFY.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
