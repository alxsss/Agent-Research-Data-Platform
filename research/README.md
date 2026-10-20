# Research workload

Agent search experiments that run on the `rdp` platform.

- `corpus/prepare.py`: stream Wikipedia, keep film/TV/music articles, write docs/sections/passages as Parquet, register them in `rdp`
- `index/build.py`: BM25 indexes over passages (setup A) and doc summaries (setup C)
- `tools.py`: the agent tool sets for setups A and C
- `try_tools.py`: call a tool by hand
