Evidence for py-econometrics/within#607. Throwaway branch, never merged.

- make_panels.py writes the 19 panels (#606's screen); run_one.py runs one solve and prints a JSON line (times, iterations, per-phase peak RSS, result checksum, the solver's own audit backward error from its PROBE_TRACE stderr, whether the coarse rung was built).
- bench.py: interleaved rotated rounds (5 below 5M rows, 3 above) -> timed.jsonl, timed-summary.txt; iters.sh/sweep.sh: deterministic counts (cal.jsonl stop calibration, sweep.jsonl 1,197 stall-rule runs); results.jsonl/summary.txt: #606's screen; bd-table.md: BlockDiagonal.
- Knobs: PROBE_CAL (1 = main's stop), PROBE_COARSE=climb|fixed, PROBE_STALL1/2=w,t, PROBE_BOTTOM=block (branch prototype/607-blockdiag).
