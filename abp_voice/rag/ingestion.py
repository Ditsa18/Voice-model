"""Document ingestion: load → chunk → embed → upsert into vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunking import chunk_text, hash_id
from .embeddings import EmbeddingModel
from .loaders import LOADERS, load
from .store import VectorStore
from ..config import get_settings
from ..languages import SUPPORTED_LANGS
from ..logging_setup import get_logger

log = get_logger(__name__)


def _detect_lang_from_filename(name: str) -> str:
    """Infer language tag from filename suffix convention (e.g. article_bn.pdf)."""
    lower = name.lower()
    for lang in SUPPORTED_LANGS:
        if f"_{lang}" in lower:
            return lang
    return "en"


def ingest_documents(
    paths: list[Path],
    store: VectorStore,
    embedder: EmbeddingModel,
    reset: bool = False,
) -> int:
    """Chunk, embed, and upsert documents. Returns total chunks indexed."""
    s = get_settings()

    if reset:
        log.info("resetting collection")
        store.reset()

    total = 0

    for path in paths:
        if path.suffix.lower() not in LOADERS:
            log.info("skip %s (unsupported extension)", path.name)
            continue

        log.info("reading %s", path)

        try:
            text = load(path)
        except Exception as e:
            log.error("failed to load %s: %s", path, e)
            continue

        chunks = chunk_text(text, s.chunk_size, s.chunk_overlap)

        if not chunks:
            log.warning("empty document: %s", path.name)
            continue

        lang = _detect_lang_from_filename(path.name)
        ids = [hash_id(c, str(path), i) for i, c in enumerate(chunks)]
        metas: list[dict[str, Any]] = [
            {"source": path.name, "path": str(path), "chunk": i, "lang": lang}
            for i in range(len(chunks))
        ]
        vecs = embedder.encode(chunks)

        store.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=vecs)
        total += len(chunks)
        log.info("+%d chunks from %s", len(chunks), path.name)

    return total
