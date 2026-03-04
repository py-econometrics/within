"""Suite registry with ``@suite`` decorator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ._types import BenchmarkResult


@dataclass
class SuiteOptions:
    """Options passed to every suite run function."""

    seed: int = 42
    tol: float = 1e-8
    maxiter: int = 2000
    quick: bool = False
    filter_problems: list[str] | None = None


@dataclass
class SuiteInfo:
    """Metadata + callable for one benchmark suite."""

    name: str
    description: str
    tags: frozenset[str]
    run_fn: Callable[[SuiteOptions], list[BenchmarkResult]]


_SUITES: dict[str, SuiteInfo] = {}


def suite(
    name: str,
    description: str = "",
    tags: tuple[str, ...] | frozenset[str] = (),
) -> Callable[
    [Callable[[SuiteOptions], list[BenchmarkResult]]],
    Callable[[SuiteOptions], list[BenchmarkResult]],
]:
    """Decorator that registers a function as a benchmark suite."""
    tag_set = frozenset(tags)

    def _decorator(
        fn: Callable[[SuiteOptions], list[BenchmarkResult]],
    ) -> Callable[[SuiteOptions], list[BenchmarkResult]]:
        if name in _SUITES:
            raise ValueError(f"Duplicate suite name: {name!r}")
        _SUITES[name] = SuiteInfo(
            name=name, description=description, tags=tag_set, run_fn=fn,
        )
        return fn

    return _decorator


def list_suites() -> dict[str, SuiteInfo]:
    """Return all registered suites."""
    return dict(_SUITES)


def get_suite(name: str) -> SuiteInfo:
    """Look up a suite by name."""
    if name not in _SUITES:
        raise KeyError(f"Unknown suite {name!r}. Available: {sorted(_SUITES)}")
    return _SUITES[name]
