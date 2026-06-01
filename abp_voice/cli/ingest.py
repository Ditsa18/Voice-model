"""Index documents into the multilingual vector store.

  python -m abp_voice ingest
  python -m abp_voice ingest <file1> <dir2> ...
  python -m abp_voice ingest --reset
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from ..config import get_settings
from ..rag.embeddings import EmbeddingModel
from ..rag.ingestion import ingest_documents
from ..rag.loaders import discover
from ..rag.store import VectorStore

console = Console()


def _collect_targets(paths: list[str]) -> list[Path]:
    s = get_settings()

    if not paths:
        return list(discover(s.data_dir))

    out: list[Path] = []

    for raw in paths:
        p = Path(raw)

        if p.is_dir():
            out.extend(discover(p))

        elif p.is_file():
            out.append(p)

        else:
            console.print(f"[red]not found[/]: {p}")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ingest documents into the vector store."
    )

    ap.add_argument("paths", nargs="*", help="files or directories to ingest")
    ap.add_argument("--reset", action="store_true", help="wipe collection first")

    args = ap.parse_args()

    targets = _collect_targets(args.paths)

    if not targets:
        s = get_settings()
        console.print(
            f"[yellow]No files. Drop .pdf/.docx/.txt/.md files in {s.data_dir} "
            "and rerun.[/]"
        )
        return 0

    if args.reset:
        console.print("[yellow]resetting collection...[/]")

    n = ingest_documents(
        paths=targets,
        store=VectorStore(),
        embedder=EmbeddingModel(),
        reset=args.reset,
    )

    console.print(
        f"[bold green]Done.[/] Indexed {n} chunks into "
        f"'{get_settings().collection_name}'."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
