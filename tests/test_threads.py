from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import within
from within import PreconditionerConfig

from conftest import as_solver_categories, every_preconditioner, generate_synthetic_data


@pytest.fixture(scope="module")
def panel():
    """60k rows: long enough that every sum splits across threads."""
    cats, _, y = generate_synthetic_data([20_000, 1_500, 300], 60_000, seed=511)
    categories = as_solver_categories(cats)
    ys = np.column_stack([y, np.sin(np.arange(y.size) * 0.1)])
    return categories, ys


@every_preconditioner
def test_a_call_returns_the_same_bits_at_any_thread_count(panel, precond) -> None:
    categories, ys = panel
    options = within.LsmrOptions(tol=1e-6, maxiter=200)
    reference = within.solve_batch(
        categories, ys, options=options, preconditioner=precond, threads=1
    )
    for threads in [None, 2]:
        r = within.solve_batch(
            categories, ys, options=options, preconditioner=precond, threads=threads
        )
        assert r.x.tobytes() == reference.x.tobytes(), threads
        assert r.demeaned.tobytes() == reference.demeaned.tobytes(), threads
        assert list(r.iterations) == list(reference.iterations), threads


def test_every_entry_point_takes_a_thread_count(panel) -> None:
    categories, ys = panel
    y = np.ascontiguousarray(ys[:, 0])
    design = within.Design(categories, threads=2)
    solver = within.Solver(
        design, preconditioner=PreconditionerConfig.Additive(), threads=2
    )
    one = solver.solve(y, threads=1)
    assert solver.solve(y, threads=2).x.tobytes() == one.x.tobytes()
    assert within.solve(categories, y, threads=2).converged
    assert all(solver.solve_batch(ys, threads=2).converged)
    m = solver.preconditioner
    x = y[: m.ncols]
    assert m.apply(x, threads=1).tobytes() == m.apply(x, threads=2).tobytes()


@pytest.mark.parametrize("threads", [0, -1])
def test_a_count_below_one_is_rejected(panel, threads) -> None:
    categories, ys = panel
    with pytest.raises(ValueError, match="threads must be at least 1"):
        within.solve(categories, np.ascontiguousarray(ys[:, 0]), threads=threads)


def test_a_count_above_the_available_parallelism_runs_on_all_of_it(panel) -> None:
    categories, ys = panel
    y = np.ascontiguousarray(ys[:, 0])
    reference = within.solve(categories, y, threads=1)
    assert (
        within.solve(categories, y, threads=10**6).x.tobytes() == reference.x.tobytes()
    )


_POOL_PROBE = textwrap.dedent(
    """
    import os, subprocess, sys
    import numpy as np
    import within

    def os_threads():
        if sys.platform.startswith("linux"):
            return len(os.listdir("/proc/self/task"))
        out = subprocess.run(
            ["ps", "-M", "-p", str(os.getpid())], capture_output=True, text=True, check=True
        )
        return len(out.stdout.splitlines()) - 1

    rng = np.random.default_rng(0)
    n = 60_000
    cats = np.asfortranarray(
        np.column_stack([np.arange(n) // 3, rng.integers(0, 1_500, n), rng.integers(0, 300, n)])
        .astype(np.uint32)
    )
    y = rng.standard_normal(n)
    diagonal = within.PreconditionerConfig.Diagonal()
    before = os_threads()
    within.solve(cats, y, preconditioner=diagonal, threads=1)
    first = os_threads()
    for _ in range(3):
        within.solve(cats, y, preconditioner=diagonal, threads=1)
    print(first - before, os_threads() - first)
    """
)


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") or sys.platform == "darwin"),
    reason="no OS thread count on this platform",
)
def test_a_call_runs_on_one_pool_of_its_count_and_reuses_it() -> None:
    """In a fresh process, `threads=1` adds one thread, once: work that escaped the budget
    would start the global pool, and a pool built per call would add one each time."""
    out = subprocess.run(
        [sys.executable, "-c", _POOL_PROBE], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["1", "0"]
