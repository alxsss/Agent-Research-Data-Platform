# Decision log: disposable vs durable

A running record of when an exploratory artifact was kept disposable and when it was promoted into the platform, and why.

| Date | Artifact | Decision | Why |
|---|---|---|---|
| 2026-10-04 | `run_accuracy`, `failure_modes`, `metric_summary` views | Durable (in `views.sql`) | Core questions for every eval run in the research plan |
| 2026-10-08 | Setup C tools | Dropped `lookup_entity` (infobox facts) for now | The HF `wikimedia/wikipedia` text has no infoboxes; C uses search_docs → open_doc → read_section instead. Revisit with Wikidata if failure analysis shows fact lookups matter |
| 2026-10-08 | Registry `_describe` | Fixed: describe a directory only if all files share one readable format | Found while registering a BM25 index (mixed .npy/.json files); the registry guessed JSON from the first file and crashed. First bug found by using the platform as a researcher |
| 2026-10-08 | Model access | OpenRouter (OpenAI-compatible API) for Claude; vLLM for the small model | One agent loop and one client for both models; switching is a base URL and model name |
| 2026-10-08 | `pytz` dependency | Added to core dependencies, with a test that lists real datasets | `rdp datasets` crashed on the first real catalog: DuckDB needs `pytz` to return TIMESTAMPTZ values. Tests had only listed an empty catalog. Second bug found by real use |
| 2026-10-08 | Corpus v1 → v2 | Rebuilt corpus after reviewing samples | v1 cut summaries at initials ("George R."), kept articles that only matched capitalized title words ("A Song of Ice and Fire"), and merged list sections such as Cast into the previous section. The registry keeps v1 and v2 side by side |
| 2026-10-18 | `prepare.py` exit | Exit with `os._exit(0)` after all work is written | Stopping the HF streaming dataset early left download threads that hung interpreter shutdown |
| 2026-10-20 | `name:latest` alias | Added to the registry (`get`, `lineage`, `register` parents) | Needed it while checking lineage: I had to know `bm25_passages:v2` existed. Asked "show me the newest" without knowing the number |
| 2026-10-20 | Schema 1.0 → 1.1 | Added nullable `episodes.stop_reason`; a test reads 1.0 and 1.1 files together | The agent loop has three outcomes (answered, ran out of steps, API error) that must be told apart in analysis. Additive change, so a minor bump with no migration |
| 2026-10-20 | `contains` grader | Kept deliberately simple for the smoke test | Good enough to see that the loop works; an LLM judge plus human grades replace it once the real eval set exists |
| 2026-10-20 | `rdp show` | Fixed crash on assistant turns with tool calls but no text | Found by the agent-loop tests; real model output often looks like this |
