"""ChromaDB persistent vector store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from ..config import get_settings


@dataclass(slots=True)
class Retrieved:
    text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "doc"))


class VectorStore:
    """Thin wrapper around a single persistent Chroma collection."""

    def __init__(self) -> None:
        s = get_settings()

        self._client = chromadb.PersistentClient(
            path=str(s.chroma_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False
            ),
        )

        self._collection_name = s.collection_name

        self._collection: Collection | None = None

    @property
    def _coll(self) -> Collection:

        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        return self._collection

    def count(self) -> int:
        return self._coll.count()

    def reset(self) -> None:

        try:
            self._client.delete_collection(
                self._collection_name
            )

        except Exception:
            pass

        self._collection = None

        _ = self._coll  # ensure collection is re-created

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:

        self._coll.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        query_embedding: list[float],
        k: int,
        lang: str | None = None,
    ) -> list[Retrieved]:

        where = None

        if lang:
            where = {
                "lang": lang
            }

        res = self._coll.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
        )

        docs = res.get("documents", [[]])[0]

        metas = res.get("metadatas", [[]])[0]

        dists = res.get("distances", [[]])[0]

        return [
            Retrieved(
                text=d,
                metadata=m or {},
                distance=dist
            )
            for d, m, dist in zip(
                docs,
                metas,
                dists
            )
        ]