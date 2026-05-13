"""Document loaders. Each returns the raw text content of a file."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path


def load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def load_docx(path: Path) -> str:
    import docx

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


LOADERS: dict[str, Callable[[Path], str]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_text,
    ".md": load_text,
}

SUPPORTED_SUFFIXES = frozenset(LOADERS.keys())


def discover(root: Path) -> Iterable[Path]:
    """Recursively yield supported files under `root`."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p


def load(path: Path) -> str:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"unsupported file type: {path}")
    return loader(path)
