from pathlib import Path


def test_required_project_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "README.md",
        root / "requirements.txt",
        root / "configs" / "baseline.yaml",
        root / "src" / "gan" / "__init__.py",
        root / "src" / "filtering" / "__init__.py",
        root / "src" / "classifier" / "__init__.py",
    ]
    assert all(path.exists() for path in required)

