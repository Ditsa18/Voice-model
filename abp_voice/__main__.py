"""Allow `python -m abp_voice <subcommand>` invocation."""

from __future__ import annotations

import sys

_SUBCOMMANDS = {"chat", "ingest", "test", "mic"}


def _print_help() -> None:
    print(
        "usage: python -m abp_voice <command> [args...]\n\n"
        "commands:\n"
        "  chat     start the continuous voice conversation\n"
        "  ingest   index documents into the vector store\n"
        "  test     one-shot call test (mic or --say)\n"
        "  mic      list audio devices or run a VU meter\n"
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in _SUBCOMMANDS:
        print(f"unknown command: {cmd}\n")
        _print_help()
        return 2

    sys.argv = [f"abp_voice {cmd}", *rest]
    if cmd == "chat":
        from .cli.chat import main as _main
    elif cmd == "ingest":
        from .cli.ingest import main as _main
    elif cmd == "test":
        from .cli.test_call import main as _main
    else:
        from .cli.mic_check import main as _main
    return int(_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
