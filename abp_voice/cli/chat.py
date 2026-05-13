"""Interactive multilingual voice conversation.

  python -m abp_voice chat                  # voice in / voice out
  python -m abp_voice chat --mode text      # type and read
  python -m abp_voice chat --seconds 6      # fixed-duration capture
  python -m abp_voice chat --lang bn        # force Whisper language
  python -m abp_voice chat --device 5       # pick mic by index
"""

from __future__ import annotations

import argparse
import sys
import traceback

import sounddevice as sd
from rich.console import Console
from rich.panel import Panel

from ..agent import VoiceAgent
from ..languages import GREETING, LANG_NAMES, SUPPORTED_LANGS, is_exit_phrase
from ..rag import RAGPipeline

console = Console()


def _text_mode(agent: VoiceAgent) -> None:
    console.print(Panel.fit("ABP Voice-RAG (TEXT mode)", style="bold cyan"))
    console.print("Type a question in English / Hindi / Bengali. 'exit' to quit.")
    while True:
        try:
            q = console.input("[bold]you[/] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        try:
            from langdetect import detect

            lang = detect(q) if detect(q) in SUPPORTED_LANGS else "en"
        except Exception:
            lang = "en"
        result, llm_ms = agent.respond(q, lang)
        console.print(
            f"[bold green]assistant[/] ({LANG_NAMES[lang]}): {result.answer}"
        )
        if result.hits:
            console.print(
                f"[dim]sources: {', '.join(result.sources)}  · llm: {llm_ms/1000:.2f}s[/]"
            )


def _voice_mode(
    agent: VoiceAgent, seconds: float | None, force_lang: str | None
) -> None:
    console.print(Panel.fit("ABP Voice-RAG (VOICE mode)", style="bold magenta"))
    console.print(
        "Speak English / Hindi / Bengali. Pause to let me respond. Ctrl+C to quit."
    )
    while True:
        try:
            console.print("[cyan]>>> press Enter and speak (Ctrl+C to exit)[/]")
            try:
                input()
            except EOFError:
                break
            turn = agent.turn(
                seconds=seconds, force_lang=force_lang, do_speak=False
            )
            if turn is None:
                console.print("[yellow](no speech detected)[/]")
                continue

            console.print(
                f"[bold]you[/] ({LANG_NAMES[turn.detected_lang]}, "
                f"conf={turn.lang_confidence:.2f}): {turn.user_text}"
            )
            if is_exit_phrase(turn.user_text):
                agent.speak(GREETING[turn.detected_lang], turn.detected_lang)
                break

            console.print(
                f"[bold green]assistant[/] ({LANG_NAMES[turn.detected_lang]}): "
                f"{turn.answer}"
            )
            if turn.sources:
                console.print(
                    f"[dim]sources: {', '.join(turn.sources)}  "
                    f"· stt: {turn.stt_ms/1000:.2f}s · llm: {turn.llm_ms/1000:.2f}s[/]"
                )
            agent.speak(turn.answer, turn.detected_lang)
        except KeyboardInterrupt:
            console.print("\n[bold]bye.[/]")
            break
        except Exception as e:
            console.print(f"[red]error:[/] {e}")
            traceback.print_exc()


def main() -> int:
    ap = argparse.ArgumentParser(description="Multilingual voice conversation.")
    ap.add_argument(
        "--mode", choices=["voice", "text"], default="voice",
        help="interaction mode (default: voice)",
    )
    ap.add_argument(
        "--seconds", type=float, default=None,
        help="record fixed duration each turn (skips silence detection)",
    )
    ap.add_argument(
        "--device", type=int, default=None,
        help="sounddevice input device index (see `abp-mic`)",
    )
    ap.add_argument(
        "--lang", choices=list(SUPPORTED_LANGS), default=None,
        help="force Whisper to transcribe in this language",
    )
    args = ap.parse_args()

    if args.device is not None:
        sd.default.device = (args.device, None)
        console.print(f"[dim]using input device #{args.device}[/]")

    agent = VoiceAgent()

    console.print("[dim]warming up models (embedding + LLM + whisper)...[/]")
    agent.warmup(include_stt=(args.mode == "voice"))

    count = RAGPipeline(store=agent.rag.store).store.count()
    if count == 0:
        console.print(
            "[yellow]Warning: knowledge base is empty. Run `python -m abp_voice ingest` "
            "(or ./ingest.py).[/]"
        )
    else:
        console.print(f"[green]Knowledge base ready: {count} chunks indexed.[/]")

    if args.mode == "voice":
        _voice_mode(agent, args.seconds, args.lang)
    else:
        _text_mode(agent)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(0)
