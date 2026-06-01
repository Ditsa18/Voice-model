"""LLM message construction and output sanitization."""

from __future__ import annotations

from ..languages import LANG_NAMES, SCRIPT_HINT, normalize_lang
from .store import Retrieved

_SYSTEM_SINGLE = (
    "You are a concise multilingual voice assistant.\n"
    "The user spoke in {lang_name}. Respond in {lang_name}.\n"
    "{script_hint}\n"
    "Use ONLY the provided context.\n"
    "If the answer is not in the context, say you don't know in {lang_name}.\n"
    "Answer using the most important facts from the context.\n"
"Prefer complete factual answers over very short fragments.\n"
"Keep the answer under 25 words.\n"
    "Output ONLY the final answer."
)

_SYSTEM_MIXED = (
    "You are a concise multilingual voice assistant.\n"
    "The user mixed languages (primarily {lang_name}).\n"
    "Reply in the same code-switched style — blend {lang_name} with English naturally.\n"
    "Example: if asked 'ABP কী? Tell me about it.' reply like "
    "'ABP একটি বড় media group, founded in 1922.'\n"
    "Use ONLY the provided context.\n"
    "If the answer is not in the context, say you don't know.\n"
    "Answer using the most important facts from the context.\n"
"Prefer complete factual answers over very short fragments.\n"
"Keep the answer under 25 words.\n"
    "Output ONLY the final answer."
)

USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"

_LEAK_PREFIXES = (
    "you are", "system:", "context:", "question:",
    "q (", "a (", "answer (", "you're", "you’re",
)


_MIXED_THRESHOLD = 0.70


def build_messages(
    question: str,
    lang: str,
    hits: list[Retrieved],
    lang_confidence: float = 1.0,
) -> list[dict[str, str]]:
    lang = normalize_lang(lang)
    lang_name = LANG_NAMES[lang]
    script_hint = SCRIPT_HINT[lang]

    is_mixed = lang_confidence < _MIXED_THRESHOLD
    template = _SYSTEM_MIXED if is_mixed else _SYSTEM_SINGLE

    context_blocks = [f"[{i}] {h.text}" for i, h in enumerate(hits, 1)]
    context = "\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"

    return [
        {
            "role": "system",
            "content": template.format(lang_name=lang_name, script_hint=script_hint),
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
