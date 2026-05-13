"""Speech synthesis: edge-tts primary, gTTS fallback.

The synthesized audio is written to a temp MP3 in `audio_cache/` and the path is returned.
Use `abp_voice.audio.play_audio_file` to play it back, or `speak()` for both.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import edge_tts

from ..audio.playback import play_audio_file
from ..config import get_settings
from ..languages import EDGE_VOICES, SUPPORTED_LANGS, normalize_lang
from ..logging_setup import get_logger

log = get_logger(__name__)


class Synthesizer:
    """Generate speech audio for short replies in EN / HI / BN."""

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── public API ────────────────────────────────────────────────────────
    def synthesize(self, text: str, lang: str) -> Path:
        lang = normalize_lang(lang)
        voice = EDGE_VOICES[lang]
        out = self._tmp_mp3()
        try:
            asyncio.run(self._edge_speak(text, voice, out))
            if not out.exists() or out.stat().st_size == 0:
                raise RuntimeError("edge-tts produced empty output")
        except Exception as e:
            log.warning("edge-tts failed (%s); falling back to gTTS", e)
            self._gtts_speak(text, lang, out)
        return out

    def speak(self, text: str, lang: str) -> None:
        if not text or not text.strip():
            return
        if lang not in SUPPORTED_LANGS:
            lang = "en"
        mp3 = self.synthesize(text, lang)
        try:
            play_audio_file(mp3)
        finally:
            mp3.unlink(missing_ok=True)

    # ── internals ─────────────────────────────────────────────────────────
    def _tmp_mp3(self) -> Path:
        return Path(
            tempfile.mktemp(suffix=".mp3", dir=str(self._settings.audio_cache))
        )

    @staticmethod
    async def _edge_speak(text: str, voice: str, out: Path) -> None:
        comm = edge_tts.Communicate(text=text, voice=voice)
        await comm.save(str(out))

    @staticmethod
    def _gtts_speak(text: str, lang: str, out: Path) -> None:
        from gtts import gTTS

        gTTS(text=text, lang=lang).save(str(out))
