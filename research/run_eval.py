"""Run a model on a question set with one search setup, logging everything to rdp.

    uv run python -m research.run_eval --arch C --model <openrouter-model-id> \\
        --questions research/questions/smoke.jsonl

Each question becomes one episode (all turns and tool calls) with a "contains"
grade: 1.0 if any gold answer appears in the model's answer after normalization.
This grader is deliberately simple; an LLM judge and human grades come later.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

from rdp import Registry, Store, check_run, start_run
from research.agent import OPENROUTER_URL, SYSTEM_PROMPT, make_client, run_agent
from research.tools import ARCHITECTURES, Corpus

DONT_KNOW = re.compile(r"\b(i )?(don'?t|do not) know\b|not (in|contained in) the collection"
                       r"|no information|cannot (find|answer)")


def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    text = "".join(ch for ch in text if ch not in set(string.punctuation) - {"'"})
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_score(answer: str | None, gold: list[str]) -> float:
    if not answer:
        return 0.0
    if any(normalize(g) == "i don't know" for g in gold):
        return float(bool(DONT_KNOW.search(answer.lower())))
    return float(any(normalize(g) in normalize(answer) for g in gold))


def load_questions(path: Path, limit: int | None) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return qs[:limit] if limit else qs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a model with one search setup.")
    p.add_argument("--arch", choices=sorted(ARCHITECTURES), required=True)
    p.add_argument("--model", required=True, help="e.g. an OpenRouter model id")
    p.add_argument("--questions", type=Path, default=Path("research/questions/smoke.jsonl"))
    p.add_argument("--limit", type=int)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--base-url", default=OPENROUTER_URL)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args(argv)

    client = make_client(args.base_url, args.api_key_env)
    store = Store()
    reg = Registry(store)
    questions = load_questions(args.questions, args.limit)
    q_version = reg.register(f"questions_{args.questions.stem}", args.questions)
    arch = ARCHITECTURES[args.arch](Corpus(args.data_dir))

    config = {
        "architecture": arch.name, "questions": q_version, "n_questions": len(questions),
        "max_steps": args.max_steps, "temperature": args.temperature,
        "base_url": args.base_url, "system_prompt": SYSTEM_PROMPT,
        "indexes": [reg.resolve("bm25_passages:latest"), reg.resolve("bm25_docs:latest")],
    }
    print(f"{arch.name} × {args.model} on {len(questions)} questions ({q_version})\n")

    total = 0.0
    with start_run("eval", model=args.model, config=config, store=store) as run:
        for q in questions:
            r = run_agent(client, args.model, arch, q["question"], args.max_steps,
                          args.temperature)
            ep = run.log_episode(
                r.turns, r.tool_calls, item_id=q["id"], architecture=arch.name,
                dataset_version=q_version, final_answer=r.final_answer,
                tokens_in=r.tokens_in, tokens_out=r.tokens_out, latency_ms=r.latency_ms,
                stop_reason=r.stop_reason)
            score = contains_score(r.final_answer, q.get("answers", []))
            run.log_grade(ep, "contains", score)
            total += score
            mark = "✓" if score else "✗"
            print(f"{mark} {q['id']}  [{len(r.tool_calls)} tool calls, {r.stop_reason}]  "
                  f"{r.final_answer!r}  (gold: {q.get('answers')})")
            print(f"    episode {ep}")

    print(f"\nscore {total:.0f}/{len(questions)}   run {run.run_id}")
    failed = [c for c in check_run(store, run.run_id) if not c.passed]
    for c in failed:
        print("data check failed:", c)
    print("\nNext:\n"
          f"  uv run rdp show <episode-id>\n"
          f"  uv run rdp sql \"SELECT * FROM run_accuracy WHERE run_id = '{run.run_id}'\"")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
