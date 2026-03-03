from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class ActivityEntry:
    source_path: str
    destination_path: str
    category_label: str
    confidence: float
    extraction_ok: bool
    timestamp_utc: str


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_activity(log_path: Path, entry: ActivityEntry) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp_utc",
                    "source_path",
                    "destination_path",
                    "category_label",
                    "confidence",
                    "extraction_ok",
                ]
            )
        writer.writerow(
            [
                entry.timestamp_utc,
                entry.source_path,
                entry.destination_path,
                entry.category_label,
                f"{entry.confidence:.6f}",
                str(entry.extraction_ok),
            ]
        )
