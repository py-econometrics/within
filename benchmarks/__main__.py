"""CLI entry point: ``python -m benchmarks``."""

from __future__ import annotations

import argparse
import sys
import json
from dataclasses import asdict, replace
from pathlib import Path

from ._framework import SuiteOptions, get_suite, list_suites
from ._table import print_pivot
from ._comparison import Comparison, print_comparison

_PROFILES: dict[str, tuple[str, tuple[str, ...], str]] = {
    "smoke": (
        "All benchmark suites at smoke scale",
        (),
        "smoke",
    ),
    "iterate": (
        "Curated fast-iteration benchmark set for backend and Auto tuning",
        (
            "scaling",
            "many_components",
            "subdomain_regimes",
            "high_fe",
        ),
        "iterate",
    ),
    "validation": (
        "Full validation sweep across all benchmark suites",
        (),
        "full",
    ),
    "auto": (
        "Additive-focused tuning set spanning many-small, few-large, and high-FE regimes",
        (
            "scaling",
            "many_components",
            "subdomain_regimes",
            "high_fe",
        ),
        "iterate",
    ),
}


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

    print("\nProfiles")
    print("-" * 80)
    for name, (description, suite_names, scale_profile) in sorted(_PROFILES.items()):
        members = ", ".join(suite_names) if suite_names else "all suites"
        print(f"{name:<14} {description}")
        print(f"{'':<14} {members}  [scale={scale_profile}]")


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

    profile_name = args.profile
    if profile_name is None:
        scale_profile = "full"
    else:
        default_names = _PROFILES[profile_name][1]
        scale_profile = _PROFILES[profile_name][2]
        if not names:
            names = sorted(suites.keys()) if not default_names else list(default_names)

    if not names:
        print("No suites selected.")
        return

    profile = scale_profile
    repeat = 1 if profile == "full" else 3
    warmup = 0 if profile == "full" else 1

    opts = SuiteOptions(
        seed=42,
        tol=1e-8,
        maxiter=2000,
        profile=profile,
        repeat=repeat,
        warmup=warmup,
    )

    label = profile_name or "custom"
    print(
        f"Running {len(names)} suite(s) with profile={label}, "
        f"scale={opts.profile}, warmup={opts.warmup}, repeat={opts.repeat}"
    )

    all_results = []
    comparisons: list[Comparison] = []
    if args.compare_reduced:
        try:
            from within._within import solve_reduced  # noqa: F401
        except ImportError as error:
            raise SystemExit(
                "Build the reduced solver first: pixi run --as-is maturin develop "
                "--uv --release --features pair-schur-prototype; "
                "then run benchmarks with pixi run --as-is."
            ) from error

    for name in names:
        info = get_suite(name)
        print(f"\n{'#' * 72}")
        print(f"# Suite: {info.name} — {info.description}")
        print(f"{'#' * 72}")
        results = info.run_fn(opts)
        all_results.extend(results)
        if args.compare_reduced:
            print(f"\n# Reduced solve: {info.name}")
            reduced = info.run_fn(replace(opts, formulation="reduced"))
            full_by_key = {(r.problem, r.config): r for r in results}
            reduced_by_key = {(r.problem, r.config): r for r in reduced}
            for key in sorted(full_by_key.keys() | reduced_by_key.keys()):
                full_result = full_by_key.get(key)
                reduced_result = reduced_by_key.get(key)
                comparisons.append(
                    Comparison(
                        suite=name,
                        problem=key[0],
                        config=key[1],
                        full=full_result,
                        reduced=reduced_result,
                    )
                )
            if args.output:
                Path(args.output).write_text(
                    json.dumps(
                        {
                            "options": asdict(opts),
                            "comparisons": [asdict(row) for row in comparisons],
                        },
                        indent=2,
                    )
                    + "\n"
                )

    if len(names) > 1 and all_results:
        print(f"\n{'=' * 72}")
        print("Combined Summary")
        print(f"{'=' * 72}")
        print_pivot(all_results)

    if args.compare_reduced:
        print_comparison(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Benchmark suite runner",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available suites")

    run_p = sub.add_parser("run", help="Run one or more suites")
    run_p.add_argument("suites", nargs="*", help="Suite names or 'all'")
    run_p.add_argument(
        "--profile",
        choices=sorted(_PROFILES),
        default=None,
        help="Named benchmark profile (suite set + scale tier)",
    )
    run_p.add_argument(
        "--compare-reduced",
        action="store_true",
        help="Compare full Schwarz and reduced PairSchur solves",
    )
    run_p.add_argument("--output", type=Path, help="Write comparison results as JSON")
    args = parser.parse_args()

    if args.command == "run" and args.output and not args.compare_reduced:
        parser.error("--output requires --compare-reduced")

    if args.command == "list":
        _cmd_list(args)
    elif args.command == "run":
        _cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
