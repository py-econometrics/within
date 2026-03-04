"""Core data types for the benchmark framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from within import CG, LSMR, GMRES


@dataclass(frozen=True)
class ProblemSpec:
    """Specification for a benchmark problem."""

    name: str
    generator: str  # Registry key in _problems.py
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for a solve via the Rust-backed API."""

    label: str
    config: LSMR | CG | GMRES


@dataclass
class BenchmarkResult:
    """Result from a single (problem, solver config) benchmark run."""

    problem: str
    config: str  # SolverConfig.label
    n_dofs: int
    n_rows: int
    setup_time: float = 0.0
    solve_time: float = 0.0
    iterations: int = 0
    final_residual: float = 0.0
    converged: bool = False
    passed: bool | None = None  # For correctness suites
    phase_timings: dict[str, float] = field(default_factory=dict)  # For profiling
    extra: dict[str, Any] = field(default_factory=dict)  # Suite-specific data
