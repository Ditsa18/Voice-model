"""faster-whisper transcription wrapper with adaptive device fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from ..config import get_settings
from ..languages import normalize_lang
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    detected_language: str
    segments: int


class Transcriber:
    """Lazy-loaded faster-whisper model with automatic CPU fallback."""

    def __init__(self) -> None:
        self._model: WhisperModel | None = None

    def _load(self) -> WhisperModel:
        if self._model is not None:
            return self._model
        s = get_settings()
        try:
            log.info(
                "loading whisper %s on %s (%s)",
                s.whisper_model, s.whisper_device, s.whisper_compute,
            )
            self._model = WhisperModel(
                s.whisper_model,
                device=s.whisper_device,
                compute_type=s.whisper_compute,
            )
        except Exception as gpu_err:
            log.warning("whisper load on %s failed (%s); falling back to CPU/int8",
                        s.whisper_device, gpu_err)
            try:
                self._model = WhisperModel(s.whisper_model, device="cpu", compute_type="int8")
            except Exception as cpu_err:
                raise RuntimeError("whisper failed to load on both GPU and CPU") from cpu_err
        return self._model

    def warmup(self) -> None:
        self._load()

    def transcribe(
        self, audio: np.ndarray, force_lang: str | None = None
    ) -> TranscriptionResult:
        if audio.size == 0:
            return TranscriptionResult("", "en", 0.0, "en", 0)

        s = get_settings()
        model = self._load()
        segments, info = model.transcribe(
            audio,
            beam_size=5,
            vad_filter=s.whisper_use_vad,
            vad_parameters={
                "threshold": s.whisper_vad_threshold,
                "min_speech_duration_ms": s.whisper_vad_min_speech_ms,
                "min_silence_duration_ms": s.whisper_vad_min_silence_ms,
            },
            language=force_lang,
            task="transcribe",
            temperature=[0.0, 0.2, 0.4, 0.6],
            condition_on_previous_text=False,
            no_speech_threshold=s.whisper_no_speech_threshold,
        )
        seg_list = list(segments)
        text = " ".join(seg.text.strip() for seg in seg_list).strip()
        detected = info.language or "en"
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        lang = normalize_lang(force_lang or detected)

        log.info(
            "whisper: detected=%s(%.2f) forced=%s segments=%d chars=%d",
            detected, prob, force_lang or "-", len(seg_list), len(text),
        )
        return TranscriptionResult(text, lang, prob, detected, len(seg_list))
