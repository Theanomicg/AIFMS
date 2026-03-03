from __future__ import annotations

from pathlib import Path


TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log", ".py", ".java", ".c", ".cpp"}


def _extract_text_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        chunks: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                chunks.append(page_text)
        return "\n".join(chunks).strip()
    except Exception:
        return ""


def _extract_text_docx(path: Path) -> str:
    try:
        import docx  # type: ignore

        doc = docx.Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception:
        return ""


def extract_text(path: Path, encoding: str = "utf-8") -> str:
    suffix = path.suffix.lower()

    if suffix in TEXT_EXTENSIONS:
        try:
            return path.read_text(encoding=encoding, errors="ignore").strip()
        except Exception:
            return ""

    if suffix == ".pdf":
        return _extract_text_pdf(path)

    if suffix == ".docx":
        return _extract_text_docx(path)

    return ""
