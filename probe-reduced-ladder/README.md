Evidence for py-econometrics/within#566 (the reduced ladder). Throwaway branch, never merged.

The prototype on top of feature/reduced-solve adds two switches: `ZZ_TB=1` turns the Schwarz rung into
term blocks (only the pairs with the eliminated term), `ZZ_TB_RUNG=1` inserts a term-block rung between
Diagonal and Schwarz in `Adaptive`; `ZZ_TRACE=1` prints each rung's stall/finish iteration.

- `make_panels.py` writes the 18 panels; `bench.py` runs the variants (3 rotated rounds, 8 threads,
  tol 1e-8) into `results.jsonl`, summarized by `analyze.py`.
- `sweep.py` runs the iteration-only stall sweep into `sweep.jsonl`, summarized by `sweep_analyze.py`.
