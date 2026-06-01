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


def detect_primary_language(text: str) -> str:
    """Detect primary language from Unicode script and explicit keywords.

    Prefers script ranges (unambiguous) over library heuristics.
    Falls back to English for Latin-only or unknown input.
    """
    lower = text.lower()

    if "in bengali" in lower or "বাংলায়" in lower:
        return "bn"
    if "in hindi" in lower or "हिंदी" in lower:
        return "hi"

    if any("ঀ" <= c <= "৿" for c in text):
        return "bn"
    if any("ऀ" <= c <= "ॿ" for c in text):
        return "hi"

    return "en"


def script_mix_confidence(text: str) -> float:
    """Return <1.0 when the text mixes multiple scripts (code-switching detected)."""
    has_latin = any("a" <= c.lower() <= "z" for c in text)
    has_devanagari = any("ऀ" <= c <= "ॿ" for c in text)
    has_bengali = any("ঀ" <= c <= "৿" for c in text)
    scripts_used = sum([has_latin, has_devanagari, has_bengali])
    return 0.5 if scripts_used > 1 else 1.0


def is_exit_phrase(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    for phrases in EXIT_PHRASES.values():
        if any(p.lower() in t for p in phrases):
            return True
    return False
