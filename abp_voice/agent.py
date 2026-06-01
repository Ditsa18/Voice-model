"""VoiceAgent: composes mic → STT → RAG → TTS for a single turn."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .audio import MicRecorder
from .languages import normalize_lang
from .logging_setup import get_logger
from .rag import RAGPipeline, RAGResult
from .stt import Transcriber
from .tts import Synthesizer

log = get_logger(__name__)


@dataclass(slots=True)
class TurnResult:
    user_text: str
    detected_lang: str
    lang_confidence: float
    answer: str
    sources: list[str]
    stt_ms: float
    llm_ms: float


class VoiceAgent:
    """Multilingual voice agent. Each `turn` is one round of speak → reply."""

    def __init__(
        self,
        rag: RAGPipeline | None = None,
        transcriber: Transcriber | None = None,
        synthesizer: Synthesizer | None = None,
        recorder: MicRecorder | None = None,
    ) -> None:
        self.rag = rag or RAGPipeline()
        self.transcriber = transcriber or Transcriber()
        self.synthesizer = synthesizer or Synthesizer()
        self.recorder = recorder or MicRecorder()

    # ── setup ─────────────────────────────────────────────────────────────
    def warmup(self, include_stt: bool = True) -> None:
        self.rag.warmup()
        if include_stt:
            self.transcriber.warmup()

    # ── single-turn helpers ───────────────────────────────────────────────
    def listen(
        self, seconds: float | None = None, force_lang: str | None = None
    ) -> tuple[str, str, float, float]:
        """Capture mic → transcribe. Returns (text, lang, confidence, elapsed_ms)."""
        t0 = perf_counter()
        if seconds and seconds > 0:
            rec = self.recorder.record_fixed(seconds)
        else:
            rec = self.recorder.record_until_silence()
        result = self.transcriber.transcribe(rec.audio, force_lang=force_lang)
        elapsed = (perf_counter() - t0) * 1000
        return result.text, result.language, result.language_probability, elapsed

    def respond(
        self, question: str, lang: str, lang_confidence: float = 1.0
    ) -> tuple[RAGResult, float]:
        t0 = perf_counter()
        result = self.rag.generate(question, normalize_lang(lang), lang_confidence)
        return result, (perf_counter() - t0) * 1000

    def speak(self, text: str, lang: str) -> None:
        self.synthesizer.speak(text, lang)

    # ── full turn ─────────────────────────────────────────────────────────
    def turn(
        self,
        seconds: float | None = None,
        force_lang: str | None = None,
        do_speak: bool = True,
    ) -> TurnResult | None:
        text, lang, conf, stt_ms = self.listen(seconds=seconds, force_lang=force_lang)
        if not text:
            return None
        rag_result, llm_ms = self.respond(text, lang, lang_confidence=conf)
        if do_speak and rag_result.answer:
            self.speak(rag_result.answer, lang)
        return TurnResult(
            user_text=text,
            detected_lang=lang,
            lang_confidence=conf,
            answer=rag_result.answer,
            sources=rag_result.sources,
            stt_ms=stt_ms,
            llm_ms=llm_ms,
        )
