#!/usr/bin/env python3
"""Tier-2 wall-clock perf alert (#147), NON-blocking.

Compares the best wall-clock time of the 1M-row 3FE solve (from the
`wallclock_1m3fe` bench harness) against a committed, FIXED reference. A
regression beyond the reference's threshold raises a GitHub job-summary alert
naming the merge. This script NEVER exits nonzero: a wall-clock signal on shared
CI runners is advisory, not a build gate. Use --update to bootstrap/refresh the
committed reference on CI hardware.
"""

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_REFERENCE = "benches/wallclock_reference.json"


def emit(markdown: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")
    print(markdown)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", default=DEFAULT_REFERENCE)
    p.add_argument(
        "--measured-ns",
        type=float,
        required=True,
        help="best wall-clock nanoseconds emitted by the wallclock_1m3fe harness",
    )
    p.add_argument("--commit", default=os.environ.get("GITHUB_SHA", "(local run)"))
    p.add_argument(
        "--update",
        action="store_true",
        help="write the measured time into the reference file and exit",
    )
    args = p.parse_args()

    ref_path = Path(args.reference)
    reference = json.loads(ref_path.read_text())
    measured = args.measured_ns
    measured_ms = measured / 1e6

    if args.update:
        reference["best_ns"] = round(measured, 3)
        ref_path.write_text(json.dumps(reference, indent=2) + "\n")
        emit(f"Updated wall-clock reference to {measured_ms:.1f} ms (best of runs).")
        return 0

    baseline = reference.get("best_ns")
    if not baseline:
        emit(
            f"⚠️ **Perf reference not bootstrapped.** Measured {measured_ms:.1f} ms. "
            f"Run the **Perf Reference** workflow via `workflow_dispatch` to "
            f"seed it on CI hardware."
        )
        return 0

    threshold_pct = float(reference.get("threshold_pct", 25.0))
    delta_pct = (measured - baseline) / baseline * 100.0
    baseline_ms = baseline / 1e6

    if delta_pct > threshold_pct:
        emit(
            f"## 🚨 Wall-clock perf regression (non-blocking)\n\n"
            f"Merge `{args.commit}` slowed the 1M-row 3FE solve.\n\n"
            f"| metric | value |\n|---|---|\n"
            f"| best (this merge) | {measured_ms:.1f} ms |\n"
            f"| reference | {baseline_ms:.1f} ms |\n"
            f"| change | +{delta_pct:.1f}% |\n"
            f"| threshold | +{threshold_pct:.1f}% |\n\n"
            f"If this cost is expected, refresh the reference via the "
            f"**Perf Reference** workflow."
        )
    else:
        emit(
            f"✅ Wall-clock within reference: {measured_ms:.1f} ms vs "
            f"{baseline_ms:.1f} ms ({delta_pct:+.1f}%, threshold +{threshold_pct:.1f}%)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
