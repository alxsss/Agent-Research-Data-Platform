"""Synthetic demo data: two eval runs, an SFT run, and a lineage chain.

All data here is randomly generated so the platform can be exercised end to end
without models or GPUs. It is not experimental evidence.
"""

from __future__ import annotations

import json
import math
import random

import pyarrow as pa
import pyarrow.parquet as pq

from rdp.contracts import check_run, no_overlap
from rdp.datasets import Registry
from rdp.metrics import start_run
from rdp.store import Store

FAILURE_MODES = ["retrieval_miss", "poor_query", "stopped_early", "over_search",
                 "evidence_ignored", "hallucination", "aggregation_error"]


def _fake_episode(run, item, architecture, rng, p_correct):
    n_calls = rng.randint(1, 4)
    turns = [{"role": "user", "content": item["question"], "token_count": 20}]
    calls = []
    for i in range(n_calls):
        turns.append({"role": "assistant", "content": f"search step {i}", "token_count": 15})
        calls.append({"turn_idx": len(turns) - 1, "tool_name": "search",
                      "args": {"query": f"{item['question']} #{i}"},
                      "result": "passage text " * rng.randint(5, 50),
                      "result_tokens": rng.randint(50, 800),
                      "duration_ms": rng.uniform(5, 120)})
        turns.append({"role": "tool", "content": f"<results {i}>", "token_count": 200})
    correct = rng.random() < p_correct
    answer = item["answer"] if correct else "unknown"
    turns.append({"role": "assistant", "content": answer, "token_count": 10})
    ep = run.log_episode(turns, calls, item_id=item["id"], architecture=architecture,
                         dataset_version=item["dataset_version"], final_answer=answer,
                         tokens_in=sum(t["token_count"] for t in turns),
                         tokens_out=10 + 15 * n_calls, latency_ms=rng.uniform(500, 8000))
    run.log_grade(ep, "exact", float(correct))
    if not correct:
        run.log_grade(ep, "human", 0.0, failure_mode=rng.choice(FAILURE_MODES))


def build_demo(store: Store, n_episodes: int = 50, seed: int = 0) -> dict:
    rng = random.Random(seed)
    reg = Registry(store)
    data = store.root / "demo_data"
    data.mkdir(exist_ok=True)

    eval_items = [{"id": f"q{i}", "question": f"Synthetic question {i}?", "answer": f"a{i}"}
                  for i in range(n_episodes)]
    eval_path = data / "evalset.jsonl"
    eval_path.write_text("".join(json.dumps(x) + "\n" for x in eval_items))
    evalset = reg.register("evalset", eval_path, description="synthetic demo eval set")
    for x in eval_items:
        x["dataset_version"] = evalset

    eval_runs = []
    for arch, p in [("A_flat_bm25", 0.45), ("C_structured", 0.65)]:
        with start_run("eval", model="reference-model", config={"architecture": arch},
                       store=store) as run:
            for item in eval_items:
                _fake_episode(run, item, arch, rng, p)
        eval_runs.append(run.run_id)

    train_items = [{"question": f"Synthetic training question {i}?",
                    "trajectory": "..."} for i in range(4 * n_episodes)]
    train_path = data / "sft.parquet"
    pq.write_table(pa.Table.from_pylist(train_items), train_path)
    sft_data = reg.register("sft_trajectories", train_path, parents=[evalset],
                            producer_run_id=eval_runs[1],
                            checks=[no_overlap(eval_path, key="question")])

    with start_run("sft", model="small-open-model", config={"lora_r": 16, "lr": 2e-4},
                   store=store) as sft:
        for step in range(0, 500, 10):
            sft.log({"train/loss": 2.5 * math.exp(-step / 150) + 0.3 + rng.gauss(0, 0.03),
                     "train/lr": 2e-4 * (1 - step / 500)}, step=step)
    ckpt = data / "checkpoint"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "adapter.bin").write_bytes(rng.randbytes(1024))
    checkpoint = reg.register("checkpoint", ckpt, parents=[sft_data], producer_run_id=sft.run_id)

    for run_id in eval_runs:
        failed = [r for r in check_run(store, run_id) if not r.passed]
        assert not failed, failed

    summary = {"eval_runs": eval_runs, "sft_run": sft.run_id, "datasets": [evalset, sft_data,
               checkpoint]}
    print(json.dumps(summary, indent=2))
    return summary
