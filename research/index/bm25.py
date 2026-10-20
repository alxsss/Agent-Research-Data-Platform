"""Thin wrapper over bm25s: build, save, load, and search with stable string IDs."""

from __future__ import annotations

import json
from pathlib import Path

import bm25s
import Stemmer

_STEMMER = Stemmer.Stemmer("english")


def _tokenize(texts: list[str], progress: bool = False):
    return bm25s.tokenize(texts, stopwords="en", stemmer=_STEMMER, show_progress=progress)


class BM25Index:
    def __init__(self, retriever: bm25s.BM25, ids: list[str]):
        self.retriever = retriever
        self.ids = ids

    @classmethod
    def build(cls, texts: list[str], ids: list[str], path: str | Path) -> BM25Index:
        retriever = bm25s.BM25()
        retriever.index(_tokenize(texts, progress=True), show_progress=True)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        retriever.save(str(path))
        (path / "ids.json").write_text(json.dumps(ids))
        return cls(retriever, ids)

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        path = Path(path)
        return cls(bm25s.BM25.load(str(path)), json.loads((path / "ids.json").read_text()))

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Top-k (id, score) pairs; empty if the query has no indexed terms."""
        tokens = bm25s.tokenize([query], stopwords="en", stemmer=_STEMMER, return_ids=False,
                                show_progress=False)
        vocab = self.retriever.vocab_dict
        terms = [t for t in tokens[0] if t in vocab]
        if not terms:
            return []
        k = min(k, len(self.ids))
        docs, scores = self.retriever.retrieve([terms], k=k, show_progress=False)
        return [(self.ids[int(i)], float(s)) for i, s in zip(docs[0], scores[0]) if s > 0]
