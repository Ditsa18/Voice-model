"""End-to-end RAG: retrieval → generation. The orchestration layer for the voice agent."""

from __future__ import annotations

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
    """Compose embeddings + vector store + LLM into a single .generate() call."""

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

    # ── high-level ────────────────────────────────────────────────────────
    def warmup(self) -> None:
        self.embeddings.warmup()
        self.store.count()  # forces collection init
        self.llm.warmup()

    def retrieve(self, query: str, k: int | None = None) -> list[Retrieved]:
        vec = self.embeddings.encode([query])[0]
        return self.store.query(vec, k or self._top_k)

    def generate(self, question: str, lang: str) -> RAGResult:
        lang = normalize_lang(lang)
        hits = self.retrieve(question)
        messages = build_messages(question, lang, hits)
        raw = self.llm.chat(messages)
        answer = scrub(raw)
        return RAGResult(answer=answer, hits=hits, lang=lang)
