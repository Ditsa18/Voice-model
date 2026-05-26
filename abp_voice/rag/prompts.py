"""LLM message construction and output sanitization."""

from __future__ import annotations

from ..languages import LANG_NAMES, SCRIPT_HINT, normalize_lang
from .store import Retrieved

SYSTEM_TEMPLATE = (
    "You are a concise multilingual voice assistant.\n"
    "ALWAYS answer in {lang_name}.\n"
    "NEVER answer in English unless the user used English.\n"
    "{script_hint}\n"
    "Use ONLY the provided context.\n"
    "If the answer is not in the context, say you don't know in {lang_name}.\n"
    "Keep the answer short and natural.\n"
    "Output ONLY the final answer."
)

USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"

_LEAK_PREFIXES = (
    "you are", "system:", "context:", "question:",
    "q (", "a (", "answer (", "you're", "you’re",
)


def build_messages(
    question: str, lang: str, hits: list[Retrieved]
) -> list[dict[str, str]]:
    lang = normalize_lang(lang)
    lang_name = LANG_NAMES[lang]
    script_hint = SCRIPT_HINT[lang]

    context_blocks = [f"[{i}] {h.text}" for i, h in enumerate(hits, 1)]
    context = "\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"

    return [
        {
            "role": "system",
            "content": SYSTEM_TEMPLATE.format(
                lang_name=lang_name, script_hint=script_hint
            ),
        },
        {
            "role": "user",
            "content": USER_TEMPLATE.format(context=context, question=question),
        },
    ]


def scrub(answer: str) -> str:
    """Strip leaked system/prompt lines from a model response."""
    if not answer:
        return answer
    cleaned: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.lower().startswith(p) for p in _LEAK_PREFIXES):
            continue
        cleaned.append(stripped)
    return " ".join(cleaned).strip() or answer.strip()
