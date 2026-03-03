from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .activity_logger import ActivityEntry, append_activity, now_utc_iso
from .classifier import ModelBundle, classify_text
from .extractor import extract_text
from .nlp import process_text
from .organizer import organize_file
from .scanner import scan_files


@dataclass(slots=True)
class PipelineSummary:
    total_files_scanned: int
    successfully_classified: int
    unclassified: int
    total_seconds: float


def run_organization_pipeline(
    input_dir: Path,
    output_dir: Path,
    model_bundle: ModelBundle,
    log_path: Path,
    recursive: bool = True,
    copy_only: bool = False,
) -> PipelineSummary:
    start = perf_counter()
    files = scan_files(input_dir, recursive=recursive)

    classified = 0
    unclassified = 0

    for file_path in files:
        raw_text = extract_text(file_path)
        extraction_ok = bool(raw_text.strip())

        base_text = raw_text if extraction_ok else file_path.stem
        clean_text = process_text(base_text)
        prediction = classify_text(model_bundle, clean_text)

        if prediction.label == "uncategorized":
            unclassified += 1
        else:
            classified += 1

        destination = organize_file(
            file_path=file_path,
            output_root=output_dir,
            category_label=prediction.label,
            copy_only=copy_only,
        )

        append_activity(
            log_path,
            ActivityEntry(
                source_path=str(file_path),
                destination_path=str(destination),
                category_label=prediction.label,
                confidence=prediction.confidence,
                extraction_ok=extraction_ok,
                timestamp_utc=now_utc_iso(),
            ),
        )

    total = perf_counter() - start
    return PipelineSummary(
        total_files_scanned=len(files),
        successfully_classified=classified,
        unclassified=unclassified,
        total_seconds=total,
    )
