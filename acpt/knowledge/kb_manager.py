"""File-backed knowledge base manager providing basic RAG hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class KnowledgeBaseError(RuntimeError):
    """Raised when knowledge base operations fail."""


class KBManager:
    """Minimal file-backed knowledge base manager for RAG descriptors."""

    def __init__(self, rag_descriptor: Dict[str, Any], storage_root: Optional[Path] = None) -> None:
        if "index_type" not in rag_descriptor:
            raise KnowledgeBaseError("RAG descriptor must include 'index_type'.")

        path_hint = rag_descriptor.get("path") or rag_descriptor.get("prefix")
        if not path_hint:
            raise KnowledgeBaseError("RAG descriptor must include 'path' or 'prefix'.")

        base_path = Path(storage_root) if storage_root is not None else Path.cwd()
        candidate = Path(path_hint)
        self._path = candidate if candidate.is_absolute() else base_path / candidate
        self._descriptor = dict(rag_descriptor)
        self._documents: List[str] = []

    @property
    def descriptor(self) -> Dict[str, Any]:
        """Return the underlying descriptor used to configure the manager."""

        return dict(self._descriptor)

    @property
    def storage_path(self) -> Path:
        """Return the filesystem path of the backing knowledge base store."""

        return self._path

    def initialize(self, seed_documents: Optional[Iterable[str]] = None) -> None:
        """Create or load the backing store with optional seed documents."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            self._documents = self._load()
        else:
            seed = list(seed_documents or ["placeholder knowledge document"])
            self._documents = [str(doc) for doc in seed]
            self._store(self._documents)

    def add_document(self, document: str) -> None:
        """Append a document to the knowledge base and persist it."""

        self._documents.append(document)
        self._store(self._documents)

    def retrieve(self, query: str, k: int = 1) -> List[str]:
        """Return up to *k* documents; placeholder ignores query and returns head documents."""

        if not self._documents and self._path.exists():
            self._documents = self._load()
        return self._documents[: max(0, k)]

    @staticmethod
    def embed(document: str) -> List[float]:
        """Return a simple embedding vector derived from document length."""

        length = float(len(document))
        return [length, length / 2.0 if length else 0.0]

    def _store(self, documents: List[str]) -> None:
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(documents, handle)

    def _load(self) -> List[str]:
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                data = json.load(handle) or []
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(f"Failed to load knowledge base: {self._path}") from exc
        return [str(item) for item in data]
