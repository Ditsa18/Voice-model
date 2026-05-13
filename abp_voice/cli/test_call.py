"""Single-call test of the voice agent.

  python -m abp_voice test                                   # mic, auto-detect, silence-stop
  python -m abp_voice test --seconds 5                       # fixed 5s recording
  python -m abp_voice test --no-tts                          # text-only reply
  python -m abp_voice test --say "..." --lang hi             # bypass mic
  python -m abp_voice test --device 4 --lang bn --seconds 6  # all the knobs
"""

from __future__ import annotations

import argparse

import sounddevice as sd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..agent import VoiceAgent
from ..config import get_settings
from ..languages import LANG_NAMES, SUPPORTED_LANGS

console = Console()


def _banner() -> None:
    s = get_settings()
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold cyan")
    t.add_column()
    t.add_row("LLM", f"{s.llm_model}  (Ollama @ {s.ollama_host})")
    t.add_row("STT", f"faster-whisper [{s.whisper_model}] on {s.whisper_device}")
    t.add_row("Embeddings", s.embedding_model)
    t.add_row("Languages", "English / Hindi / Bengali (auto-detected)")
    console.print(Panel(t, title="ABP Voice-RAG · call test", border_style="cyan"))


def _resolve_say_lang(text: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        from langdetect import detect

        d = detect(text)
        return d if d in SUPPORTED_LANGS else "en"
    except Exception:
        return "en"


def main() -> int:
    ap = argparse.ArgumentParser(description="Single call test.")
    ap.add_argument("--no-tts", action="store_true", help="skip speaking the reply")
    ap.add_argument("--say", default=None, help="skip mic; use this text instead")
    ap.add_argument(
        "--lang", choices=list(SUPPORTED_LANGS), default=None,
        help=(
            "with --say it sets the typed-text language; "
            "without --say it forces Whisper's transcription language"
        ),
    )
    ap.add_argument(
        "--seconds", type=float, default=None,
        help="record fixed duration (skips silence detection)",
    )
    ap.add_argument(
        "--device", type=int, default=None,
        help="sounddevice input device index",
    )
    args = ap.parse_args()

    if args.device is not None:
        sd.default.device = (args.device, None)
        console.print(f"[dim]using input device #{args.device}[/]")

    _banner()

    agent = VoiceAgent()
    if agent.rag.store.count() == 0:
        console.print(
            "[yellow]Knowledge base is empty. Run `python -m abp_voice ingest` first.[/]"
        )
        return 1

    if args.say:
        lang = _resolve_say_lang(args.say, args.lang)
        console.print(
            f"[bold]you[/] (forced, {LANG_NAMES[lang]}): {args.say}"
        )
        rag_result, llm_ms = agent.respond(args.say, lang)
        console.print(
            f"\n[bold green]assistant[/] ({LANG_NAMES[lang]}):\n{rag_result.answer}"
        )
        if rag_result.hits:
            console.print(
                f"[dim]sources: {', '.join(rag_result.sources)}  · llm: {llm_ms/1000:.2f}s[/]"
            )
        if not args.no_tts:
            console.print("[dim]speaking reply...[/]")
            agent.speak(rag_result.answer, lang)
            console.print("[dim](done)[/]")
        return 0

    # Mic path
    if args.seconds:
        console.print(
            f"[cyan]>>> press Enter, then speak for ~{args.seconds:.0f} seconds[/]"
        )
    else:
        console.print("[cyan]>>> press Enter, then speak (pause ~1s when done)[/]")
    try:
        input()
    except EOFError:
        return 0

    turn = agent.turn(
        seconds=args.seconds, force_lang=args.lang, do_speak=not args.no_tts
    )
    if turn is None:
        console.print("[yellow](no speech detected)[/]")
        return 0

    console.print(
        f"[bold]you said[/] -> [magenta]{turn.user_text}[/]\n"
        f"[dim]detected language: [bold]{LANG_NAMES[turn.detected_lang]} ({turn.detected_lang})[/]  "
        f"confidence: {turn.lang_confidence:.2f}  · stt: {turn.stt_ms/1000:.2f}s[/]"
    )
    console.print(
        f"\n[bold green]assistant[/] ({LANG_NAMES[turn.detected_lang]}):\n{turn.answer}"
    )
    if turn.sources:
        console.print(
            f"[dim]sources: {', '.join(turn.sources)}  · llm: {turn.llm_ms/1000:.2f}s[/]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
