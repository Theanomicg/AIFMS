from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(slots=True)
class PreviewItem:
    source_path: Path
    predicted_label: str
    confidence: float
    extraction_ok: bool


@dataclass(slots=True)
class OrganizationAssignment:
    source_path: Path
    category_label: str
    confidence: float
    extraction_ok: bool


def preview_classification_pipeline(
    input_dir: Path,
    model_bundle: ModelBundle,
    recursive: bool = True,
) -> list[PreviewItem]:
    files = scan_files(input_dir, recursive=recursive)
    results: list[PreviewItem] = []

    for file_path in files:
        raw_text = extract_text(file_path)
        extraction_ok = bool(raw_text.strip())
        base_text = raw_text if extraction_ok else file_path.stem
        clean_text = process_text(base_text)
        prediction = classify_text(model_bundle, clean_text)
        results.append(
            PreviewItem(
                source_path=file_path,
                predicted_label=prediction.label,
                confidence=prediction.confidence,
                extraction_ok=extraction_ok,
            )
        )

    return results


def apply_organization_assignments(
    assignments: list[OrganizationAssignment],
    output_dir: Path,
    log_path: Path,
    copy_only: bool = False,
) -> PipelineSummary:
    from time import perf_counter

    start = perf_counter()
    classified = 0
    unclassified = 0

    for item in assignments:
        if not item.source_path.exists():
            continue

        if item.category_label == "uncategorized":
            unclassified += 1
        else:
            classified += 1

        destination = organize_file(
            file_path=item.source_path,
            output_root=output_dir,
            category_label=item.category_label,
            copy_only=copy_only,
        )

        append_activity(
            log_path,
            ActivityEntry(
                source_path=str(item.source_path),
                destination_path=str(destination),
                category_label=item.category_label,
                confidence=item.confidence,
                extraction_ok=item.extraction_ok,
                timestamp_utc=now_utc_iso(),
            ),
        )

    total = perf_counter() - start
    return PipelineSummary(
        total_files_scanned=len(assignments),
        successfully_classified=classified,
        unclassified=unclassified,
        total_seconds=total,
    )


def run_organization_pipeline(
    input_dir: Path,
    output_dir: Path,
    model_bundle: ModelBundle,
    log_path: Path,
    recursive: bool = True,
    copy_only: bool = False,
) -> PipelineSummary:
    preview = preview_classification_pipeline(
        input_dir=input_dir,
        model_bundle=model_bundle,
        recursive=recursive,
    )
    assignments = [
        OrganizationAssignment(
            source_path=item.source_path,
            category_label=item.predicted_label,
            confidence=item.confidence,
            extraction_ok=item.extraction_ok,
        )
        for item in preview
    ]
    return apply_organization_assignments(
        assignments=assignments,
        output_dir=output_dir,
        log_path=log_path,
        copy_only=copy_only,
    )
