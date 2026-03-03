from __future__ import annotations

from pathlib import Path


def scan_files(root_dir: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [p for p in root_dir.glob(pattern) if p.is_file()]
    return sorted(files)
