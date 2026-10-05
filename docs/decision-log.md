# Decision log: disposable vs durable

A running record of when an exploratory artifact was kept disposable and when it was promoted into the platform, and why.

| Date | Artifact | Decision | Why |
|---|---|---|---|
| 2026-10-04 | `run_accuracy`, `failure_modes`, `metric_summary` views | Durable (in `views.sql`) | Core questions for every eval run in the research plan |
