"""CLI entry point: ``python -m benchmarks``."""

from __future__ import annotations

import argparse
import sys

from ._registry import SuiteOptions, get_suite, list_suites
from ._table import print_pivot, print_table


def _cmd_list(args: argparse.Namespace) -> None:
    suites = list_suites()
    if not suites:
        print("No suites registered.")
        return
    name_w = max(len(n) for n in suites) + 2
    print(f"{'Suite':<{name_w}} {'Tags':<30} Description")
    print("-" * 80)
    for name, info in sorted(suites.items()):
        tags = ", ".join(sorted(info.tags)) or "-"
        print(f"{name:<{name_w}} {tags:<30} {info.description}")


def _cmd_run(args: argparse.Namespace) -> None:
    suites = list_suites()
    names: list[str] = []

    if "all" in args.suites:
        names = sorted(suites.keys())
    else:
        for s in args.suites:
            if s in suites:
                names.append(s)
            else:
                print(f"Unknown suite: {s!r}", file=sys.stderr)
                sys.exit(1)

    if args.tag:
        tag_set = set(args.tag)
        names = [n for n in names if suites[n].tags & tag_set]

    if not names:
        print("No suites selected.")
        return

    opts = SuiteOptions(
        seed=args.seed,
        tol=args.tol,
        maxiter=args.maxiter,
        quick=args.quick,
    )

    all_results = []
    for name in names:
        info = get_suite(name)
        print(f"\n{'#' * 72}")
        print(f"# Suite: {info.name} — {info.description}")
        print(f"{'#' * 72}")
        results = info.run_fn(opts)
        all_results.extend(results)

    if len(names) > 1 and all_results:
        print(f"\n{'=' * 72}")
        print("Combined Summary")
        print(f"{'=' * 72}")
        print_pivot(all_results)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Benchmark suite runner",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available suites")

    run_p = sub.add_parser("run", help="Run one or more suites")
    run_p.add_argument("suites", nargs="+", help="Suite names or 'all'")
    run_p.add_argument("--quick", action="store_true", help="Use small problems")
    run_p.add_argument("--tag", nargs="+", help="Filter by tag(s)")
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--tol", type=float, default=1e-8)
    run_p.add_argument("--maxiter", type=int, default=2000)

    args = parser.parse_args()

    if args.command == "list":
        _cmd_list(args)
    elif args.command == "run":
        _cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
