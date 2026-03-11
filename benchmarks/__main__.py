"""CLI entry point: ``python -m benchmarks``."""

from __future__ import annotations

import argparse
import sys

from ._framework import SuiteOptions, get_suite, list_suites
from ._table import print_pivot

_PRESETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "iterate": (
        "Curated fast-iteration benchmark set for backend and Auto tuning",
        (
            "verify",
            "preconditioners_3fe",
            "ac_comparison",
            "scaling",
            "many_components",
            "high_fe",
            "akm_panel",
            "fixest_comparison",
        ),
    ),
    "validation": (
        "Full validation sweep across all benchmark suites",
        (),
    ),
    "auto": (
        "Additive-focused tuning set spanning many-small, few-large, and high-FE regimes",
        (
            "verify",
            "scaling",
            "many_components",
            "high_fe",
            "akm_panel",
            "fixest_comparison",
        ),
    ),
    "local_solver": (
        "Local-solver and ApproxChol variant comparisons",
        (
            "ac_comparison",
            "graph_backend_comparison",
        ),
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

    print("\nPresets")
    print("-" * 80)
    for name, (description, suite_names) in sorted(_PRESETS.items()):
        members = ", ".join(suite_names) if suite_names else "all suites"
        print(f"{name:<14} {description}")
        print(f"{'':<14} {members}")


def _cmd_run(args: argparse.Namespace) -> None:
    suites = list_suites()
    names: list[str] = []

    if args.preset:
        if args.preset not in _PRESETS:
            print(f"Unknown preset: {args.preset!r}", file=sys.stderr)
            sys.exit(1)
        preset_names = _PRESETS[args.preset][1]
        names = sorted(suites.keys()) if not preset_names else list(preset_names)
    elif "all" in args.suites:
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

    profile = "smoke" if args.quick else args.profile
    repeat = args.repeat
    warmup = args.warmup
    if repeat is None:
        repeat = 1 if profile == "full" else 3
    if warmup is None:
        warmup = 0 if profile == "full" else 1

    opts = SuiteOptions(
        seed=args.seed,
        tol=args.tol,
        maxiter=args.maxiter,
        profile=profile,
        repeat=repeat,
        warmup=warmup,
    )

    print(
        f"Running {len(names)} suite(s) with profile={opts.profile}, "
        f"warmup={opts.warmup}, repeat={opts.repeat}"
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
    run_p.add_argument("suites", nargs="*", help="Suite names or 'all'")
    run_p.add_argument(
        "--preset",
        choices=sorted(_PRESETS),
        help="Run a named benchmark preset instead of explicit suites",
    )
    run_p.add_argument("--quick", action="store_true", help="Use small problems")
    run_p.add_argument(
        "--profile",
        choices=("smoke", "iterate", "full"),
        default="full",
        help="Benchmark profile: smoke, iterate, or full",
    )
    run_p.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Number of timed repeats per case (default: profile-dependent)",
    )
    run_p.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Number of warmup solves per case (default: profile-dependent)",
    )
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
