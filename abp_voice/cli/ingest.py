"""Index documents into the multilingual vector store.

  python -m abp_voice ingest                    # ingest everything in ./data
  python -m abp_voice ingest <file1> <dir2> ... # ingest specific paths
  python -m abp_voice ingest --reset            # wipe & rebuild
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from ..config import get_settings
from ..rag.chunking import chunk_text, hash_id
from ..rag.embeddings import EmbeddingModel
from ..rag.loaders import LOADERS, discover, load
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


def _ingest(targets: list[Path], reset: bool) -> int:
    s = get_settings()
    store = VectorStore()
    if reset:
        console.print("[yellow]resetting collection...[/]")
        store.reset()

    embedder = EmbeddingModel()
    total_chunks = 0

    for path in targets:
        if path.suffix.lower() not in LOADERS:
            console.print(f"[yellow]skip[/] {path} (unsupported)")
            continue
        console.print(f"[cyan]reading[/] {path}")
        try:
            text = load(path)
        except Exception as e:
            console.print(f"[red]failed[/] {path}: {e}")
            continue

        chunks = chunk_text(text, s.chunk_size, s.chunk_overlap)
        if not chunks:
            console.print(f"[yellow]empty[/] {path}")
            continue

        ids = [hash_id(c, str(path), i) for i, c in enumerate(chunks)]
        metas = [
            {"source": path.name, "path": str(path), "chunk": i}
            for i in range(len(chunks))
        ]
        vecs = embedder.encode(chunks)
        store.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=vecs)
        total_chunks += len(chunks)
        console.print(f"  [green]+{len(chunks)} chunks[/]")

    return total_chunks


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest documents into the vector store.")
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

    n = _ingest(targets, reset=args.reset)
    console.print(
        f"[bold green]Done.[/] Indexed {n} chunks into "
        f"'{get_settings().collection_name}'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
