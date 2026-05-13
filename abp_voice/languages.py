"""Language constants: supported codes, display names, TTS voices, and script hints."""

from __future__ import annotations

from typing import Final

SUPPORTED_LANGS: Final[tuple[str, ...]] = ("en", "hi", "bn")

LANG_NAMES: Final[dict[str, str]] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
}

EDGE_VOICES: Final[dict[str, str]] = {
    "en": "en-US-AriaNeural",
    "hi": "hi-IN-SwaraNeural",
    "bn": "bn-IN-TanishaaNeural",
}

SCRIPT_HINT: Final[dict[str, str]] = {
    "en": "Use the Latin (English) alphabet only.",
    "hi": (
        "Use the Devanagari script only (देवनागरी). "
        "Do NOT use Bengali or Latin letters."
    ),
    "bn": (
        "Use the Bengali script only (বাংলা লিপি). "
        "Do NOT use Devanagari or Latin letters."
    ),
}

GREETING: Final[dict[str, str]] = {
    "en": "Goodbye!",
    "hi": "अलविदा!",
    "bn": "বিদায়!",
}

EXIT_PHRASES: Final[dict[str, set[str]]] = {
    "en": {"exit", "quit", "stop", "goodbye", "bye"},
    "hi": {"बंद करो", "रुको", "अलविदा", "बंद"},
    "bn": {"বন্ধ করো", "থামো", "বিদায়", "শেষ"},
}


def normalize_lang(code: str | None) -> str:
    """Clamp a Whisper-detected language code to our supported set; default to English."""
    if code and code in SUPPORTED_LANGS:
        return code
    return "en"


def is_exit_phrase(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    for phrases in EXIT_PHRASES.values():
        if any(p.lower() in t for p in phrases):
            return True
    return False
