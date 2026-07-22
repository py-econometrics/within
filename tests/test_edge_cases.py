"""Edge case tests for the within fixed-effects solver.

Covers: NaN/Inf input rejection, degenerate problems, non-contiguous arrays,
low maxiter, and config boundary values. Pathological-weight tolerance
(zero/negative/tiny/huge) is covered with strong oracles on the Rust side
(edge_cases.rs); the binding only needs the NaN/Inf validation contract here.
"""

from __future__ import annotations

import numpy as np
import pytest

from within import LsmrOptions, PreconditionerConfig, Solver, solve


def as_solver_categories(cats):
    return np.asfortranarray(np.column_stack(cats).astype(np.uint32))


# ---------------------------------------------------------------------------
# NaN / Inf validation (binding contract)
# ---------------------------------------------------------------------------


class TestNanInfPropagation:
    def test_nan_in_y_rejected(self):
        """NaN in y should be rejected at LsmrOptions input validation."""
        cats = as_solver_categories([np.array([0, 1, 0]), np.array([0, 0, 1])])
        y = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="finite"):
            solve(cats, y)

    def test_inf_in_y_rejected(self):
        """Inf in y should be rejected at LsmrOptions input validation."""
        cats = as_solver_categories([np.array([0, 1, 0]), np.array([0, 0, 1])])
        y = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="finite"):
            solve(cats, y)


# ---------------------------------------------------------------------------
# Degenerate y vectors
# ---------------------------------------------------------------------------


class TestDegenerateY:
    def test_all_zero_y_gives_zero_solution(self):
        """y = 0 implies x = 0; solver should converge immediately."""
        cats = as_solver_categories(
            [np.array([0, 1, 0, 1, 2]), np.array([0, 0, 1, 1, 0])]
        )
        y = np.zeros(5)
        result = solve(cats, y)
        assert result.converged
        np.testing.assert_allclose(result.x, 0.0, atol=1e-10)

    def test_constant_y_demeaned_is_zero(self):
        """A constant response has zero within-group variation; demeaned should be ~0."""
        rng = np.random.RandomState(42)
        cats = as_solver_categories(
            [rng.randint(0, 10, size=100), rng.randint(0, 10, size=100)]
        )
        y = np.full(100, 5.0)
        result = solve(cats, y)
        if result.converged:
            np.testing.assert_allclose(result.demeaned, 0.0, atol=1e-4)

    def test_all_same_category_assignments(self):
        """All observations in one group per factor leaves no variation to explain."""
        cats = np.zeros((5, 2), dtype=np.uint32, order="F")
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = solve(cats, y, preconditioner=PreconditionerConfig.Off)
        assert np.all(np.isfinite(result.x))


# ---------------------------------------------------------------------------
# Solver configuration limits
# ---------------------------------------------------------------------------


class TestSolverConfigLimits:
    def test_maxiter_1_produces_finite_result(self):
        """maxiter=1 terminates after one iteration and returns finite x."""
        rng = np.random.default_rng(42)
        cats = as_solver_categories(
            [rng.integers(0, 50, size=1000), rng.integers(0, 50, size=1000)]
        )
        y = rng.standard_normal(1000)
        result = solve(cats, y, options=LsmrOptions(maxiter=1, tol=1e-15))
        assert not result.converged
        assert np.all(np.isfinite(result.x))

    def test_maxiter_2_not_converged_on_hard_problem(self):
        """Extremely tight tolerance + 2 iterations should not converge."""
        rng = np.random.default_rng(42)
        cats = as_solver_categories(
            [rng.integers(0, 50, size=1000), rng.integers(0, 50, size=1000)]
        )
        y = rng.standard_normal(1000)
        result = solve(cats, y, options=LsmrOptions(maxiter=2, tol=1e-15))
        assert not result.converged
        assert np.all(np.isfinite(result.x))

    def test_tight_tolerance_does_not_crash(self):
        """tol=1e-14 should not crash; convergence depends on the problem."""
        cats = as_solver_categories(
            [np.array([0, 1, 0, 1, 2]), np.array([0, 0, 1, 1, 0])]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = solve(cats, y, options=LsmrOptions(tol=1e-14))
        assert np.all(np.isfinite(result.x))

    def test_tol_1_converges_immediately(self):
        """tol=1.0 (very loose) should converge in very few iterations."""
        rng = np.random.default_rng(0)
        cats = as_solver_categories(
            [rng.integers(0, 20, size=200), rng.integers(0, 20, size=200)]
        )
        y = rng.standard_normal(200)
        result = solve(cats, y, options=LsmrOptions(tol=1.0))
        assert result.converged
        assert result.iterations <= 5


# ---------------------------------------------------------------------------
# Minimal / degenerate problem sizes
# ---------------------------------------------------------------------------


class TestMinimalProblemSizes:
    def test_two_factor_levels_each_converges(self):
        """Small but well-connected problem with two levels per factor."""
        cats = as_solver_categories([np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = solve(cats, y)
        assert result.converged
        assert np.all(np.isfinite(result.x))


# ---------------------------------------------------------------------------
# Non-contiguous input arrays (binding contract)
# ---------------------------------------------------------------------------


class TestNonContiguousInputs:
    def test_non_contiguous_y_gives_same_result_as_contiguous(self):
        """Non-contiguous y (slice of larger array) should give the same result."""
        cats = as_solver_categories(
            [np.array([0, 1, 0, 1, 2]), np.array([0, 0, 1, 1, 0])]
        )
        y_direct = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Build non-contiguous view with same values
        y_full = np.zeros(10, dtype=np.float64)
        y_full[::2] = y_direct
        y_strided = y_full[::2]
        assert not y_strided.flags["C_CONTIGUOUS"]

        result_contiguous = solve(cats, y_direct)
        result_strided = solve(cats, y_strided)

        if result_contiguous.converged and result_strided.converged:
            np.testing.assert_allclose(
                result_contiguous.x, result_strided.x, atol=1e-12
            )

    def test_non_contiguous_weights_gives_same_result_as_contiguous(self):
        """Non-contiguous weights should give the same result as contiguous."""
        cats = as_solver_categories(
            [np.array([0, 1, 0, 1, 2]), np.array([0, 0, 1, 1, 0])]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w_direct = np.array([0.5, 1.0, 1.5, 0.8, 1.2])

        w_full = np.zeros(10, dtype=np.float64)
        w_full[::2] = w_direct
        w_strided = w_full[::2]
        assert not w_strided.flags["C_CONTIGUOUS"]

        result_contiguous = solve(cats, y, weights=w_direct)
        result_strided = solve(cats, y, weights=w_strided)

        if result_contiguous.converged and result_strided.converged:
            np.testing.assert_allclose(
                result_contiguous.x, result_strided.x, atol=1e-12
            )

    def test_c_contiguous_categories_emits_warning(self):
        """C-contiguous (row-major) categories should emit a UserWarning."""
        cats_c = np.array(
            [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.uint32, order="C"
        )
        assert cats_c.flags["C_CONTIGUOUS"]
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.warns(UserWarning, match="F-contiguous"):
            solve(cats_c, y)


# ---------------------------------------------------------------------------
# PreconditionerConfig.Off (no preconditioner)
# ---------------------------------------------------------------------------


class TestNoPreconditioner:
    def test_preconditioner_off_converges_on_easy_problem(self):
        """Unpreconditioned LsmrOptions should still converge on a simple problem."""
        rng = np.random.default_rng(99)
        cats = as_solver_categories(
            [rng.integers(0, 10, size=200), rng.integers(0, 10, size=200)]
        )
        y = rng.standard_normal(200)
        result = solve(cats, y, preconditioner=PreconditionerConfig.Off)
        assert result.converged
        assert np.all(np.isfinite(result.x))

    def test_preconditioner_off_result_finite(self):
        """Without preconditioner, result should at least be finite."""
        cats = as_solver_categories(
            [np.array([0, 1, 0, 1, 2]), np.array([0, 0, 1, 1, 0])]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = solve(cats, y, preconditioner=PreconditionerConfig.Off)
        assert np.all(np.isfinite(result.x))

    def test_solver_preconditioner_none_returns_none(self):
        """Solver built with PreconditionerConfig.Off should return None from preconditioner()."""
        cats = as_solver_categories(
            [np.array([0, 1, 0, 1, 2]), np.array([0, 0, 1, 1, 0])]
        )
        solver = Solver(cats, preconditioner=PreconditionerConfig.Off)
        assert solver.preconditioner is None
