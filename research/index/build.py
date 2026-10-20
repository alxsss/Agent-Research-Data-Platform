"""Build the BM25 indexes the search tools use.

    uv run python -m research.index.build

Reads <data-dir>/corpus/*.parquet and writes:
    <data-dir>/index/passages   BM25 over "title | section | passage"  (setup A)
    <data-dir>/index/docs       BM25 over "title | summary"            (setup C)
Each index is registered in the rdp registry with the corpus table it came from
as its parent.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

from rdp import Registry, Store
from research.index.bm25 import BM25Index


def build_indexes(data_dir: Path, store: Store | None = None) -> dict:
    corpus, out = data_dir / "corpus", data_dir / "index"
    built = {}
    specs = {
        "passages": ("passage_id", lambda r: f"{r['title']} | {r['section']}\n{r['text']}"),
        "docs": ("doc_id", lambda r: f"{r['title']} | {r['summary']}"),
    }
    for name, (id_col, to_text) in specs.items():
        start = time.time()
        rows = pq.read_table(corpus / f"{name}.parquet").to_pylist()
        BM25Index.build([to_text(r) for r in rows], [r[id_col] for r in rows], out / name)
        print(f"indexed {len(rows):,} {name} in {time.time() - start:.0f}s -> {out / name}")
        built[name] = out / name

    if store is not None:
        reg = Registry(store)
        for name, path in built.items():
            parent = reg.register(f"wiki_{name}", corpus / f"{name}.parquet")  # existing version
            print("registered:", reg.register(f"bm25_{name}", path, parents=[parent]))
    return built


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build BM25 indexes over the corpus.")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args(argv)
    build_indexes(Path(args.data_dir), Store())


if __name__ == "__main__":
    main()
