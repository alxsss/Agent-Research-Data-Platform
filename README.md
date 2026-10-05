# Agent Research Data Platform

> **Status: in progress (Week 1 of 4).** The platform core works and is tested. The research workload (agent search experiments and fine-tuning) is not built yet. All data produced by `rdp demo` is **synthetic**.

A small research data platform for LLM agent experiments. It gives researchers one trusted place for **transcripts, training metrics, eval grades and datasets with lineage**, stored as Parquet and queried with DuckDB.

It is built with its first user in mind: a companion research project comparing knowledge-base architectures for search agents and fine-tuning a small model to use them (see [`docs/design.md`](docs/design.md)).

## What's here

| Component | What it does |
|---|---|
| `rdp.schemas` | Versioned Arrow schemas for `runs`, `metrics`, `episodes`, `turns`, `tool_calls`, `grades` |
| `rdp.store` | Append-only, atomically written Parquet partitions; content-addressed blob store for large payloads |
| `rdp.metrics` | `start_run()` / `run.log()` / `run.log_episode()` / `run.log_grade()`, with buffered writes and crash-safe status; Hugging Face/TRL callback |
| `rdp.datasets` | Dataset registry: versions, content hashes, row counts, schemas, lineage (DuckDB catalog) |
| `rdp.contracts` | Run checks (referential integrity, contiguity, token sanity, schema version) and dataset checks (train/eval contamination) |
| `rdp.query` | DuckDB views over the store, plus curated research views (`run_accuracy`, `failure_modes`, `metric_summary`) |
| `rdp` CLI | `runs`, `compare`, `check`, `datasets`, `lineage`, `show`, `sql`, `demo` |

## Quickstart

```bash
uv sync
uv run rdp demo                      # synthetic runs, metrics, datasets
uv run rdp runs
uv run rdp sql "SELECT * FROM run_accuracy ORDER BY run_id, grader"
uv run rdp lineage checkpoint:v1
uv run pytest
```

```python
import rdp

with rdp.start_run("eval", model="my-model", config={"architecture": "C"}) as run:
    ep = run.log_episode(
        turns=[{"role": "user", "content": "Who directed Alien?"},
               {"role": "assistant", "content": "Ridley Scott"}],
        tool_calls=[{"turn_idx": 1, "tool_name": "search", "args": {"q": "Alien director"},
                     "result": "..."}],
        architecture="C", tokens_in=120, tokens_out=8)
    run.log_grade(ep, grader="exact", score=1.0)

con = rdp.connect(rdp.Store())
con.sql("SELECT * FROM run_accuracy").show()
```

## Design principles
- **One canonical transcript model** that research code writes into directly, instead of per-team copies.
- **Append-only and immutable.** Status changes and re-grades are new rows; views pick the latest; history is kept for audit.
- **Trust is checked, not assumed.** Contracts run on write and on demand; registration fails if a check fails.
- **Reproducibility.** Every run records its config and git commit; every dataset records its content hash, producer run and parents.

See [`docs/design.md`](docs/design.md) for the full design and [`docs/adr/`](docs/adr/) for key decisions.

## Roadmap
- [x] Week 1: schemas, store, metrics library, registry + lineage, contracts, CLI, tests, CI
- [ ] Week 2: research harness (corpus, indexes, agent loop, eval set) writing through `rdp`
- [ ] Week 3: experiments and LoRA SFT logged through `rdp.metrics`
- [ ] Week 4: scale benchmarks (synthetic), schema v2 migration, write-up
