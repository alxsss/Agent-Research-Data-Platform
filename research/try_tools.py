"""Call a search tool by hand, the way the agent will.

    uv run python -m research.try_tools A search "who directed alien"
    uv run python -m research.try_tools C search_docs "ridley scott science fiction"
    uv run python -m research.try_tools C open_doc '{"title": "Alien (film)"}'
    uv run python -m research.try_tools C read_section '{"title": "Alien (film)", "section": "Production"}'

A plain-text argument is treated as {"query": ...}.
"""

from __future__ import annotations

import argparse
import json
import time

from research.tools import ARCHITECTURES, Corpus


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Call a search tool by hand.")
    p.add_argument("arch", choices=sorted(ARCHITECTURES))
    p.add_argument("tool")
    p.add_argument("args", help='JSON object, or plain text used as {"query": ...}')
    p.add_argument("--data-dir", default="data")
    a = p.parse_args(argv)

    start = time.time()
    arch = ARCHITECTURES[a.arch](Corpus(a.data_dir))
    print(f"(loaded corpus and indexes in {time.time() - start:.1f}s)\n")
    args = a.args if a.args.lstrip().startswith("{") else json.dumps({"query": a.args})
    print(arch.call(a.tool, args))


if __name__ == "__main__":
    main()
