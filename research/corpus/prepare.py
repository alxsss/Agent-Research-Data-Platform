"""Build the film/TV/music Wikipedia corpus.

    uv run python -m research.corpus.prepare --max-docs 30000

Streams the Hugging Face ``wikimedia/wikipedia`` dump (no full download), keeps
articles whose opening sentence says they are about a film, TV show, album, song,
band, or person in those fields, and writes three Parquet tables to
``<data-dir>/corpus/``:

    docs.parquet      one row per article: title, url, summary, section outline
    sections.parquet  one row per section: heading and text
    passages.parquet  ~150-word chunks within sections (the unit setup A searches)

Each table is registered as a dataset version in the rdp registry, with lineage
docs -> sections -> passages.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rdp import Registry, Store

SNAPSHOT = "20231101.en"

# Case-sensitive on purpose: lowercase topic words only, so capitalized title words
# ("A Song of Ice and Fire", "The Band") don't count.
TOPIC = re.compile(
    r"\b(?:is|was|are|were)\b[^.]{0,120}?\b("
    r"film|movie|television series|TV series|sitcom|miniseries|soap opera|"
    r"animated series|web series|album|song|soundtrack|band|singer|songwriter|"
    r"rapper|musician|composer|actor|actress|filmmaker|film director|record producer"
    r")s?\b"
)
SKIP_TITLE = re.compile(r"^(List of|Lists of|Index of|Outline of)\b|\(disambiguation\)")
SKIP_SECTIONS = {"references", "external links", "see also", "notes", "further reading",
                 "bibliography", "sources", "citations", "footnotes"}
# Headings that are usually followed by a list rather than prose.
LIST_HEADINGS = {"cast", "main cast", "voice cast", "track listing", "personnel",
                 "filmography", "selected filmography", "discography", "awards",
                 "awards and nominations", "accolades", "charts", "weekly charts",
                 "certifications", "release history", "band members", "members", "credits",
                 "episodes", "soundtrack", "songs"}
LEAD = "(lead)"
END_PUNCT = tuple('.!?:;,"”)]')

PASSAGE_TARGET_WORDS = 150
PASSAGE_MAX_WORDS = 220


def is_topic(text: str) -> bool:
    return bool(TOPIC.search(text[:400]))


def _is_heading(line: str, next_line: str) -> bool:
    # The HF dump keeps headings as bare short lines; a heading is followed by prose,
    # unless it is a known list heading such as "Cast" or "Track listing".
    s = line.strip()
    if not (0 < len(s) <= 80 and len(s.split()) <= 8 and s[0].isupper()
            and not s.endswith(END_PUNCT)):
        return False
    return s.lower() in LIST_HEADINGS or len(next_line.strip()) >= 60


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split article text into (heading, text) pairs; the first is the lead."""
    lines = text.split("\n")
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    next_of = dict(itertools.pairwise(nonblank))
    sections, heading, buf = [], LEAD, []
    for i, line in enumerate(lines):
        nxt = lines[next_of[i]] if i in next_of else ""
        if i > 0 and _is_heading(line, nxt):
            sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line.strip(), []
        else:
            buf.append(line)
    sections.append((heading, "\n".join(buf).strip()))
    return [(h, t) for h, t in sections if t and h.lower() not in SKIP_SECTIONS]


def chunk(text: str) -> list[str]:
    """Greedy paragraph packing into ~PASSAGE_TARGET_WORDS-word passages."""
    words_out, chunks = [], []
    for para in (p for p in re.split(r"\n+", text) if p.strip()):
        words = para.split()
        while len(words) > PASSAGE_MAX_WORDS:  # split very long paragraphs
            head, words = words[:PASSAGE_TARGET_WORDS], words[PASSAGE_TARGET_WORDS:]
            if words_out:
                chunks.append(" ".join(words_out))
                words_out = []
            chunks.append(" ".join(head))
        if words_out and len(words_out) + len(words) > PASSAGE_MAX_WORDS:
            chunks.append(" ".join(words_out))
            words_out = []
        words_out.extend(words)
        if len(words_out) >= PASSAGE_TARGET_WORDS:
            chunks.append(" ".join(words_out))
            words_out = []
    if words_out:
        chunks.append(" ".join(words_out))
    return chunks


# A sentence ends at . ! ? followed by a capital, but not after an initial
# ("George R. R. Martin") or a common abbreviation ("Jr.", "Dr.", "St.").
SENTENCE_END = re.compile(
    r"(?<!\b[A-Z]\.)(?<!\b(?:Jr|Sr|Dr|Mr|Ms|St|vs|Co)\.)(?<!\bMrs\.)(?<!\bInc\.)"
    r"(?<=[.!?])\s+(?=[A-Z])"
)


def first_sentence(text: str, limit: int = 300) -> str:
    s = SENTENCE_END.split(text.strip(), maxsplit=1)[0]
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + "…"


def article_rows(article: dict) -> tuple[dict, list[dict], list[dict]] | None:
    """Turn one HF article into (doc, sections, passages) rows, or None to skip."""
    sections = split_sections(article["text"])
    if not sections or sections[0][0] != LEAD:
        return None
    doc_id, title = str(article["id"]), article["title"]
    doc = {"doc_id": doc_id, "title": title, "url": article["url"],
           "summary": first_sentence(sections[0][1]),
           "outline": [h for h, _ in sections], "n_words": len(article["text"].split())}
    sec_rows, pas_rows = [], []
    for s_idx, (heading, text) in enumerate(sections):
        sec_rows.append({"doc_id": doc_id, "section_idx": s_idx, "heading": heading,
                         "text": text})
        for c_idx, passage in enumerate(chunk(text)):
            pas_rows.append({"passage_id": f"{doc_id}-{s_idx}-{c_idx}", "doc_id": doc_id,
                             "title": title, "section": heading, "text": passage})
    return doc, sec_rows, pas_rows


def write_corpus(articles, out_dir: Path, max_docs: int, log_every: int = 50_000) -> dict:
    docs, sections, passages = [], [], []
    scanned, start = 0, time.time()
    for article in articles:
        scanned += 1
        if scanned % log_every == 0:
            print(f"  scanned {scanned:,} articles, kept {len(docs):,} "
                  f"({time.time() - start:.0f}s)", flush=True)
        if SKIP_TITLE.search(article["title"]) or not is_topic(article["text"]):
            continue
        rows = article_rows(article)
        if rows is None:
            continue
        docs.append(rows[0])
        sections.extend(rows[1])
        passages.extend(rows[2])
        if len(docs) >= max_docs:
            break
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rows in [("docs", docs), ("sections", sections), ("passages", passages)]:
        paths[name] = out_dir / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), paths[name], compression="zstd")
    print(f"scanned {scanned:,}; kept {len(docs):,} docs, {len(sections):,} sections, "
          f"{len(passages):,} passages -> {out_dir}")
    return paths


def register(paths: dict, store: Store, description: str) -> dict:
    reg = Registry(store)
    docs = reg.register("wiki_docs", paths["docs"], description=description)
    secs = reg.register("wiki_sections", paths["sections"], parents=[docs],
                        description=description)
    pas = reg.register("wiki_passages", paths["passages"], parents=[secs],
                       description=description)
    return {"docs": docs, "sections": secs, "passages": pas}


def print_samples(docs_path: Path, n: int = 3, seed: int = 0) -> None:
    docs = pq.read_table(docs_path).to_pylist()
    for d in random.Random(seed).sample(docs, min(n, len(docs))):
        print(f"\n{d['title']}  ({d['url']})\n  {d['summary']}\n  sections: {d['outline']}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--max-docs", type=int, default=30_000)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--snapshot", default=SNAPSHOT)
    args = p.parse_args(argv)

    from datasets import load_dataset

    print(f"streaming wikimedia/wikipedia {args.snapshot} ...", flush=True)
    articles = load_dataset("wikimedia/wikipedia", args.snapshot, split="train", streaming=True)
    paths = write_corpus(articles, Path(args.data_dir) / "corpus", args.max_docs)
    versions = register(paths, Store(), f"wikimedia/wikipedia {args.snapshot}, film/TV/music "
                        f"filter, max_docs={args.max_docs}")
    print("registered:", versions)
    print_samples(paths["docs"])


if __name__ == "__main__":
    main()
    # Stopping a streaming HF dataset early leaves background download threads alive,
    # and they can block interpreter shutdown (seen as a hang after "Retrying in 1s").
    # Everything is written and registered by now, so exit without waiting for them.
    sys.stdout.flush()
    os._exit(0)
