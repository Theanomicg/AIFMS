from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FileRecord:
    path: Path
    extracted_text: str
    clean_text: str


@dataclass(slots=True)
class ClassificationResult:
    label: str
    confidence: float
