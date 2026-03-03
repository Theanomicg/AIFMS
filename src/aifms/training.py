from __future__ import annotations

import csv
from pathlib import Path

from .nlp import process_text


def load_training_dataset(csv_path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "text" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            raise ValueError("Dataset CSV must include 'text' and 'label' columns.")
        for row in reader:
            raw_text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if raw_text and label:
                texts.append(process_text(raw_text))
                labels.append(label)
    if not texts:
        raise ValueError("No valid rows found in dataset.")
    return texts, labels
