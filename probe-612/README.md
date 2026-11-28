Evidence for py-econometrics/within#612. Throwaway branch, never merged.

- build.sh base|cand: release wheels of base (f93b6527) and this branch into venv-base / venv-cand.
- bench_all.sh: every timed comparison, interleaved rotated rounds (5 below 4M rows, 3 at 4M and above) via tools/bench.py -> *.jsonl, progress in *.log; summary.txt from tools/summ.py.
- r.sh: one run with knobs. Panels: probe-607's make_panels / tools/make_scaled.py (same draws).
- Knobs: PROBE_SPACE=components|components-lift|rigid-assembled|rigid-small|rigid-small-lift|sa|agg|geneo, PROBE_SA_THETA, PROBE_SA_SMOOTH, PROBE_SA_SMALL, PROBE_SA_STRENGTH, PROBE_GENEO_THETA/TAU/MAX, PROBE_RECYCLE=k (batch), PROBE_RITZ=path (Ritz values of the first LSMR pass).
