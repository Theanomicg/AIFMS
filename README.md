# AI-Based File Management System (AIFMS)

Offline desktop-oriented pipeline for semantic file organization:

1. Scan files from an input directory
2. Extract text from supported files
3. Preprocess text with lightweight NLP
4. Classify content using TF-IDF + ML model
5. Move files into category folders
6. Log all actions and produce a summary report

## Project Layout

```text
AIFMS/
  src/aifms/
    cli.py
    scanner.py
    extractor.py
    nlp.py
    classifier.py
    organizer.py
    activity_logger.py
    pipeline.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Train a Model

Training data CSV format:

```csv
text,label
"invoice for march payment","finance"
"operating systems assignment 2","academic"
```

Run training:

```bash
python -m aifms.cli train --dataset data/train.csv --model-out models/aifms_model.joblib
```

Quick start with included sample:

```bash
python -m aifms.cli train --dataset data/sample_train.csv --model-out models/aifms_model.joblib
```

## Organize Files

```bash
python -m aifms.cli organize ^
  --input-dir sample_input ^
  --output-dir organized ^
  --model-path models/aifms_model.joblib ^
  --log-path logs/run_log.csv
```

## Notes

- Runs fully offline.
- Text extraction for `pdf` and `docx` is optional and depends on installed libraries.
- If extraction fails, the system falls back to filename-based classification.

Optional extractor dependencies:

```bash
pip install pypdf python-docx
```
