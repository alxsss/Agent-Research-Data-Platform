"""Search tools the agent can call, one tool set per knowledge-base architecture.

Setup A ("flat"):        search(query) -> top passages
Setup C ("structured"):  search_docs(query) -> titles + one-line summaries
                         open_doc(title)     -> summary + section outline
                         read_section(title, section) -> section text

Tool schemas use the OpenAI function-calling format, which both OpenRouter and
vLLM accept. Tools never raise on bad input; they return an explanation the
agent can act on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from research.index.bm25 import BM25Index

SECTION_CHAR_LIMIT = 6000


def _schema(name: str, description: str, **params: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object",
                       "properties": {p: {"type": "string", "description": d}
                                      for p, d in params.items()},
                       "required": list(params)}}}


class Corpus:
    def __init__(self, data_dir: str | Path = "data"):
        data_dir = Path(data_dir)
        c = data_dir / "corpus"
        self.docs = {r["doc_id"]: r for r in pq.read_table(c / "docs.parquet").to_pylist()}
        self.by_title = {d["title"].lower(): i for i, d in self.docs.items()}
        self.sections: dict[str, list[tuple[str, str]]] = {}
        for r in pq.read_table(c / "sections.parquet").to_pylist():
            self.sections.setdefault(r["doc_id"], []).append((r["heading"], r["text"]))
        self.passages = {r["passage_id"]: r for r in pq.read_table(
            c / "passages.parquet", columns=["passage_id", "title", "section", "text"]
        ).to_pylist()}
        self.passage_index = BM25Index.load(data_dir / "index" / "passages")
        self.doc_index = BM25Index.load(data_dir / "index" / "docs")

    def find_doc(self, title: str) -> str | None:
        return self.by_title.get(title.strip().lower())

    def suggest_titles(self, title: str, k: int = 3) -> list[str]:
        return [self.docs[i]["title"] for i, _ in self.doc_index.search(title, k)]


class Architecture:
    name: str
    tools: tuple[dict, ...]

    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def call(self, tool: str, args: dict | str) -> str:
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                return f"Error: arguments are not valid JSON: {args!r}"
        fn = getattr(self, f"tool_{tool}", None)
        if fn is None or tool not in {t["function"]["name"] for t in self.tools}:
            return f"Error: unknown tool {tool!r}."
        try:
            return fn(**args)
        except TypeError as e:
            return f"Error: bad arguments for {tool}: {e}"


class FlatBM25(Architecture):
    """Setup A: one search over ~150-word passages."""

    name = "A_flat_bm25"
    k = 5
    tools = (_schema("search", "Keyword search over Wikipedia passages about films, TV and "
                     "music. Returns the top 5 passages with their article and section.",
                     query="Search query"),)

    def tool_search(self, query: str) -> str:
        hits = self.corpus.passage_index.search(query, self.k)
        if not hits:
            return "No results."
        out = []
        for n, (pid, _) in enumerate(hits, 1):
            p = self.corpus.passages[pid]
            out.append(f"[{n}] {p['title']} — {p['section']}\n{p['text']}")
        return "\n\n".join(out)


class Structured(Architecture):
    """Setup C: navigate from documents to outlines to sections."""

    name = "C_structured"
    k = 8
    tools = (
        _schema("search_docs", "Find Wikipedia articles about films, TV and music. Returns "
                "up to 8 article titles, each with a one-sentence summary.",
                query="Search query"),
        _schema("open_doc", "Open an article by exact title. Returns its summary and the "
                "list of its sections.", title="Exact article title from search_docs"),
        _schema("read_section", "Read one section of an article.",
                title="Exact article title", section="Section name from open_doc"),
    )

    def tool_search_docs(self, query: str) -> str:
        hits = self.corpus.doc_index.search(query, self.k)
        if not hits:
            return "No results."
        return "\n".join(f"- {self.corpus.docs[i]['title']}: {self.corpus.docs[i]['summary']}"
                         for i, _ in hits)

    def _missing(self, title: str) -> str:
        sugg = self.corpus.suggest_titles(title)
        return f"No article titled {title!r}." + (f" Did you mean: {sugg}?" if sugg else "")

    def tool_open_doc(self, title: str) -> str:
        doc_id = self.corpus.find_doc(title)
        if doc_id is None:
            return self._missing(title)
        d = self.corpus.docs[doc_id]
        return (f"{d['title']}\nSummary: {d['summary']}\nSections:\n"
                + "\n".join(f"- {h}" for h in d["outline"]))

    def tool_read_section(self, title: str, section: str) -> str:
        doc_id = self.corpus.find_doc(title)
        if doc_id is None:
            return self._missing(title)
        secs = self.corpus.sections.get(doc_id, [])
        for heading, text in secs:
            if heading.lower() == section.strip().lower():
                more = "\n[section truncated]" if len(text) > SECTION_CHAR_LIMIT else ""
                return text[:SECTION_CHAR_LIMIT] + more
        return f"No section {section!r} in {title!r}. Sections: {[h for h, _ in secs]}"


ARCHITECTURES = {"A": FlatBM25, "C": Structured}
