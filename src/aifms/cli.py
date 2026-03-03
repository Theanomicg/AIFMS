from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .classifier import load_model, save_model, train_model
from .nlp import process_text
from .pipeline import run_organization_pipeline


def _load_training_dataset(csv_path: Path) -> tuple[list[str], list[str]]:
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


def cmd_train(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    model_out = Path(args.model_out)
    texts, labels = _load_training_dataset(dataset)
    bundle = train_model(texts, labels, min_confidence=args.min_confidence)
    save_model(bundle, model_out)
    print(f"Model trained and saved to: {model_out}")
    print(f"Samples: {len(texts)}")
    return 0


def cmd_organize(args: argparse.Namespace) -> int:
    model_bundle = load_model(Path(args.model_path))
    summary = run_organization_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_bundle=model_bundle,
        log_path=Path(args.log_path),
        recursive=not args.no_recursive,
        copy_only=args.copy_only,
    )

    print("AIFMS Run Summary")
    print(f"- Total files scanned: {summary.total_files_scanned}")
    print(f"- Successfully classified: {summary.successfully_classified}")
    print(f"- Unclassified: {summary.unclassified}")
    print(f"- Total processing time (s): {summary.total_seconds:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-Based File Management System (AIFMS)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a text classification model")
    train_parser.add_argument("--dataset", required=True, help="CSV path with text,label columns")
    train_parser.add_argument("--model-out", required=True, help="Output model artifact path (.joblib)")
    train_parser.add_argument("--min-confidence", type=float, default=0.40, help="Threshold for category assignment")
    train_parser.set_defaults(func=cmd_train)

    organize_parser = subparsers.add_parser("organize", help="Run file organization pipeline")
    organize_parser.add_argument("--input-dir", required=True, help="Directory to scan")
    organize_parser.add_argument("--output-dir", required=True, help="Directory where organized folders are created")
    organize_parser.add_argument("--model-path", required=True, help="Trained model artifact path")
    organize_parser.add_argument("--log-path", required=True, help="CSV log output path")
    organize_parser.add_argument("--copy-only", action="store_true", help="Copy files instead of moving")
    organize_parser.add_argument("--no-recursive", action="store_true", help="Disable recursive scan")
    organize_parser.set_defaults(func=cmd_organize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
