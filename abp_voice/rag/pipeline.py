"""End-to-end RAG: retrieval → generation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ..config import get_settings
from ..languages import normalize_lang
from ..logging_setup import get_logger
from .embeddings import EmbeddingModel
from .llm import OllamaLLM
from .prompts import build_messages, scrub
from .store import Retrieved, VectorStore

log = get_logger(__name__)


@dataclass(slots=True)
class RAGResult:
    answer: str
    hits: list[Retrieved]
    lang: str

    @property
    def sources(self) -> list[str]:
        return [h.source for h in self.hits]


class RAGPipeline:
    """Compose embeddings + vector store + LLM."""

    def __init__(
        self,
        store: VectorStore | None = None,
        embeddings: EmbeddingModel | None = None,
        llm: OllamaLLM | None = None,
    ) -> None:

        self.store = store or VectorStore()

        self.embeddings = embeddings or EmbeddingModel()

        self.llm = llm or OllamaLLM()

        self._top_k = get_settings().top_k

    # Warmup
    def warmup(self) -> None:

        self.embeddings.warmup()

        self.store.count()

        self.llm.warmup()

    # Retrieval
    def retrieve(
        self,
        query: str,
        lang: str | None = None,
        k: int | None = None,
    ) -> list[Retrieved]:
        vec = self.embeddings.encode([query])[0]
        return self.store.query(vec, k or self._top_k, lang=lang)

    # Full generation
    def generate(
        self,
        question: str,
        lang: str,
        lang_confidence: float = 1.0,
    ) -> RAGResult:

        lang = normalize_lang(lang)

        hits = self.retrieve(question, lang=lang)

        messages = build_messages(
            question,
            lang,
            hits,
            lang_confidence=lang_confidence,
        )

        raw = self.llm.chat(messages)

        answer = scrub(raw)

        return RAGResult(
            answer=answer,
            hits=hits,
            lang=lang
        )

    # Streaming generation
    def stream_generate(
        self,
        question: str,
        lang: str,
    ) -> tuple[Iterator[str], list[Retrieved]]:

        lang = normalize_lang(lang)

        hits = self.retrieve(question, lang=lang)

        messages = build_messages(question, lang, hits)

        return self.llm.stream_chat(messages), hits