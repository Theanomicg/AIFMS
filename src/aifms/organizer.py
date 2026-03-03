from __future__ import annotations

import shutil
from pathlib import Path


def organize_file(file_path: Path, output_root: Path, category_label: str, copy_only: bool = False) -> Path:
    category_dir = output_root / category_label
    category_dir.mkdir(parents=True, exist_ok=True)
    destination = category_dir / file_path.name

    if copy_only:
        shutil.copy2(file_path, destination)
    else:
        shutil.move(str(file_path), str(destination))
    return destination
