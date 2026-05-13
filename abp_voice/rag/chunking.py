"""Sentence-aware text chunking for multilingual content."""

from __future__ import annotations

import hashlib
import re

# Break candidates ordered by preference (paragraph > period > Devanagari/Bengali full stop).
_BREAK_CANDIDATES = ("\n\n", ". ", "। ", "? ", "! ")


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    text = normalize_whitespace(text)
    if not text:
        return []

    chunks: list[str] = []
    i, n = 0, len(text)
    while i < n:
        end = min(n, i + size)
        slice_ = text[i:end]
        last_break = max(slice_.rfind(b) for b in _BREAK_CANDIDATES)
        if last_break > size * 0.5 and end < n:
            end = i + last_break + 1
            slice_ = text[i:end]
        chunks.append(slice_.strip())
        if end >= n:
            break
        i = end - overlap
    return [c for c in chunks if c]


def hash_id(text: str, source: str, idx: int) -> str:
    payload = f"{source}|{idx}|{text[:64]}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:24]
