"""Sentence-transformers multilingual embeddings."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from ..config import get_settings
from ..logging_setup import get_logger

log = get_logger(__name__)


class EmbeddingModel:
    """Lazy-loaded multilingual encoder."""

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        name = get_settings().embedding_model
        log.info("loading embedding model: %s", name)
        self._model = SentenceTransformer(name)
        return self._model

    def warmup(self) -> None:
        self._load()

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vecs = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.tolist()
