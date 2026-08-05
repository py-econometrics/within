from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
from numpy.typing import NDArray

import within


EXPECT_FREE_THREADED = os.environ.get("WITHIN_EXPECT_FREE_THREADED") == "1"
pytestmark = pytest.mark.skipif(
    not EXPECT_FREE_THREADED,
    reason="requires a free-threaded CPython build",
)


def _is_gil_enabled() -> bool:
    probe = getattr(sys, "_is_gil_enabled", None)
    assert callable(probe)
    return bool(probe())


def test_import_keeps_gil_disabled() -> None:
    assert not _is_gil_enabled()


def test_shared_solver_handles_parallel_solves() -> None:
    assert not _is_gil_enabled()

    rng = np.random.default_rng(42)
    categories = np.asfortranarray(
        rng.integers(0, 32, size=(4_096, 3), dtype=np.uint32)
    )
    y = rng.standard_normal(categories.shape[0])
    y.setflags(write=False)

    solver = within.Solver(categories)
    expected = solver.solve(y)
    assert expected.converged

    barrier = Barrier(4)

    def solve(_: int) -> NDArray[np.float64]:
        barrier.wait()
        result = solver.solve(y)
        assert result.converged
        return np.asarray(result.x)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(solve, range(16)))

    for result in results:
        np.testing.assert_allclose(result, expected.x)
