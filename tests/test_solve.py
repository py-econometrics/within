from __future__ import annotations

import numpy as np
import pytest

from within import (
    CG,
    GMRES,
    AdditiveSchwarz,
    MultiplicativeSchwarz,
    OperatorRepr,
    solve,
)

from conftest import generate_synthetic_data


def as_solver_categories(cats):
    return np.column_stack(cats).astype(np.uintp)


@pytest.fixture()
def problem():
    """Two-factor problem with 50 levels each, 10k observations."""
    np.random.seed(42)
    cats = [
        np.random.randint(0, 50, size=10_000),
        np.random.randint(0, 50, size=10_000),
    ]
    y = np.random.randn(10_000)
    return cats, y


class TestSolveDefaults:
    def test_default_config(self, problem):
        cats, y = problem
        result = solve(as_solver_categories(cats), y)
        assert result.converged
        assert result.iterations > 0
        assert result.residual < 1e-6

    def test_explicit_cg(self, problem):
        cats, y = problem
        result = solve(
            as_solver_categories(cats), y, CG(operator=OperatorRepr.Explicit)
        )
        assert result.converged

    def test_gmres(self, problem):
        cats, y = problem
        result = solve(
            as_solver_categories(cats), y, GMRES(preconditioner=MultiplicativeSchwarz())
        )
        assert result.converged

    def test_rejects_multiplicative_cg(self, problem):
        cats, y = problem
        with pytest.raises(
            TypeError, match="CG preconditioner must be AdditiveSchwarz or None"
        ):
            solve(
                as_solver_categories(cats),
                y,
                CG(preconditioner=MultiplicativeSchwarz()),
            )


class TestSolveWeighted:
    def test_weighted(self, problem):
        cats, y = problem
        weights = np.random.exponential(1.0, size=len(y))
        result = solve(as_solver_categories(cats), y, weights=weights)
        assert result.converged


class TestNLevelsInference:
    def test_inferred_matches_explicit(self, problem):
        cats, y = problem
        categories = as_solver_categories(cats)
        result_inferred = solve(categories, y)
        result_explicit = solve(categories, y, n_levels=[50, 50])
        np.testing.assert_allclose(result_inferred.x, result_explicit.x, atol=1e-10)

    def test_explicit_n_levels(self, problem):
        cats, y = problem
        result = solve(as_solver_categories(cats), y, n_levels=[50, 50])
        assert result.converged


class TestPreconditioners:
    def test_additive_schwarz(self, problem):
        cats, y = problem
        result = solve(
            as_solver_categories(cats), y, CG(preconditioner=AdditiveSchwarz())
        )
        assert result.converged

    def test_multiplicative_schwarz(self, problem):
        cats, y = problem
        result = solve(
            as_solver_categories(cats), y, GMRES(preconditioner=MultiplicativeSchwarz())
        )
        assert result.converged


class TestDemean:
    def test_recovers_fixed_effects(self):
        """Solve D x = y where y = D x_true, and check we recover x_true."""
        n_levels = [50, 50]
        cats, x_true, y = generate_synthetic_data(n_levels, 10_000, seed=7)
        result = solve(as_solver_categories(cats), y)
        assert result.converged
        # x is identified only up to a global constant per factor; compare demeaned vectors
        x_hat = result.x
        offset = 0
        for n in n_levels:
            block_hat = x_hat[offset : offset + n]
            block_true = x_true[offset : offset + n]
            np.testing.assert_allclose(
                block_hat - block_hat.mean(),
                block_true - block_true.mean(),
                atol=1e-4,
            )
            offset += n

    def test_residual_is_orthogonal_to_design(self):
        """The residual y - D x should be orthogonal to every column of D."""
        np.random.seed(99)
        n_obs, n_levels = 5_000, [30, 40]
        cats = [np.random.randint(0, nl, size=n_obs) for nl in n_levels]
        y = np.random.randn(n_obs)
        result = solve(as_solver_categories(cats), y)
        residual = y.copy()
        offset = 0
        for f, nl in enumerate(n_levels):
            for lvl in range(nl):
                residual[cats[f] == lvl] -= result.x[offset + lvl]
            offset += nl
        # D^T r should be ~0
        offset = 0
        for f, nl in enumerate(n_levels):
            for lvl in range(nl):
                dot = residual[cats[f] == lvl].sum()
                assert abs(dot) < 1e-4, f"factor {f}, level {lvl}: D^T r = {dot}"
            offset += nl


class TestSolveResult:
    def test_result_fields(self, problem):
        cats, y = problem
        result = solve(as_solver_categories(cats), y)
        assert isinstance(result.x, np.ndarray)
        assert result.x.dtype == np.float64
        assert len(result.x) == 100  # 50 + 50 levels
        assert result.time_total >= 0
        assert result.time_setup >= 0
        assert result.time_solve >= 0


class TestGenerateSyntheticData:
    def test_basic(self):
        cats, x_true, y = generate_synthetic_data([100, 100], 10_000)
        assert len(cats) == 2
        assert all(len(c) == 10_000 for c in cats)
        assert x_true.shape == (200,)
        assert y.shape == (10_000,)

    def test_deterministic(self):
        c1, x1, y1 = generate_synthetic_data([50, 50], 5_000, seed=123)
        c2, x2, y2 = generate_synthetic_data([50, 50], 5_000, seed=123)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
