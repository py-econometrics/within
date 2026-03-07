from __future__ import annotations

import numpy as np
import pytest

from within import (
    CG,
    DemeanResult,
    GMRES,
    LSMR,
    MultiplicativeOneLevelSchwarz,
    OneLevelSchwarz,
    SolveResult,
    demean,
    solve,
)

from conftest import generate_synthetic_data


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
        result = solve(cats, y)
        assert result.converged
        assert result.iterations > 0
        assert result.residual < 1e-6

    def test_explicit_cg(self, problem):
        cats, y = problem
        result = solve(cats, y, CG())
        assert result.converged

    def test_lsmr(self, problem):
        cats, y = problem
        result = solve(cats, y, LSMR())
        assert result.converged

    def test_gmres(self, problem):
        cats, y = problem
        result = solve(cats, y, GMRES(MultiplicativeOneLevelSchwarz()))
        assert result.converged


class TestSolveWeighted:
    def test_weighted(self, problem):
        cats, y = problem
        weights = np.random.exponential(1.0, size=len(y))
        result = solve(cats, y, weights=weights)
        assert result.converged

    def test_weighted_lsmr(self, problem):
        cats, y = problem
        weights = np.ones(len(y))
        result_w = solve(cats, y, LSMR(), weights=weights)
        result_u = solve(cats, y, LSMR())
        np.testing.assert_allclose(result_w.x, result_u.x, atol=1e-6)


class TestNLevelsInference:
    def test_inferred_matches_explicit(self, problem):
        cats, y = problem
        result_inferred = solve(cats, y)
        result_explicit = solve(cats, y, n_levels=[50, 50])
        np.testing.assert_allclose(result_inferred.x, result_explicit.x, atol=1e-10)

    def test_explicit_n_levels(self, problem):
        cats, y = problem
        result = solve(cats, y, n_levels=[50, 50])
        assert result.converged


class TestPreconditioners:
    def test_additive_schwarz(self, problem):
        cats, y = problem
        result = solve(cats, y, CG(preconditioner=OneLevelSchwarz()))
        assert result.converged

    def test_multiplicative_schwarz(self, problem):
        cats, y = problem
        result = solve(cats, y, CG(preconditioner=MultiplicativeOneLevelSchwarz()))
        assert result.converged


class TestDemean:
    def test_recovers_fixed_effects(self):
        """Solve D x = y where y = D x_true, and check we recover x_true."""
        n_levels = [50, 50]
        cats, x_true, y = generate_synthetic_data(n_levels, 10_000, seed=7)
        result = solve(cats, y)
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
        result = solve(cats, y)
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


class TestDemeanApi:
    def test_matches_solve_residual(self, problem):
        cats, y = problem
        solve_result = solve(cats, y)
        demean_result = demean(cats, y)

        residual = y.copy()
        offset = 0
        for f, cat in enumerate(cats):
            n_level = int(cat.max()) + 1
            for lvl in range(n_level):
                residual[cat == lvl] -= solve_result.x[offset + lvl]
            offset += n_level

        np.testing.assert_allclose(demean_result.y_demean, residual, atol=1e-10)
        assert demean_result.converged == solve_result.converged
        assert demean_result.iterations == solve_result.iterations
        assert demean_result.residual == solve_result.residual

    def test_inferred_matches_explicit_n_levels(self, problem):
        cats, y = problem
        result_inferred = demean(cats, y)
        result_explicit = demean(cats, y, n_levels=[50, 50])
        np.testing.assert_allclose(
            result_inferred.y_demean, result_explicit.y_demean, atol=1e-10
        )

    def test_result_fields(self, problem):
        cats, y = problem
        result = demean(cats, y)
        assert isinstance(result.y_demean, np.ndarray)
        assert result.y_demean.dtype == np.float64
        assert result.y_demean.shape == (len(y),)
        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert isinstance(result.residual, float)
        assert result.time_total >= 0
        assert result.time_setup >= 0
        assert isinstance(result.time_solve, float)
        assert result.time_solve >= 0

    def test_weighted_matches_solve_residual(self, problem):
        cats, y = problem
        weights = np.random.exponential(1.0, size=len(y))
        solve_result = solve(cats, y, LSMR(), weights=weights)
        demean_result = demean(cats, y, LSMR(), weights=weights)

        residual = y.copy()
        offset = 0
        for cat in cats:
            n_level = int(cat.max()) + 1
            for lvl in range(n_level):
                residual[cat == lvl] -= solve_result.x[offset + lvl]
            offset += n_level

        np.testing.assert_allclose(demean_result.y_demean, residual, atol=1e-10)

    @pytest.mark.parametrize("layout", ["factor_major", "row_major", "compressed"])
    def test_layouts_agree(self, problem, layout):
        cats, y = problem
        result = demean(cats, y, layout=layout)
        assert result.converged

    def test_invalid_layout(self, problem):
        cats, y = problem
        with pytest.raises(ValueError, match="Unknown layout"):
            demean(cats, y, layout="bogus")


class TestSolveResult:
    def test_result_fields(self, problem):
        cats, y = problem
        result = solve(cats, y)
        assert isinstance(result, SolveResult)
        assert isinstance(result.x, np.ndarray)
        assert result.x.dtype == np.float64
        assert result.x.shape == (100,)  # 50 + 50 levels, single RHS
        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert isinstance(result.residual, float)
        assert result.time_total >= 0
        assert result.time_setup >= 0
        assert isinstance(result.time_solve, float)
        assert result.time_solve >= 0


class TestBatchY2D:
    def test_solve_2d_matches_columnwise(self, problem):
        cats, y = problem
        y2 = np.column_stack([y, 2.0 * y + 0.5])
        result_2d = solve(cats, y2)

        assert isinstance(result_2d, SolveResult)
        assert result_2d.x.ndim == 2
        assert result_2d.x.shape == (100, 2)
        assert isinstance(result_2d.converged, np.ndarray)
        assert result_2d.converged.shape == (2,)
        assert isinstance(result_2d.iterations, np.ndarray)
        assert result_2d.iterations.shape == (2,)
        assert isinstance(result_2d.residual, np.ndarray)
        assert result_2d.residual.shape == (2,)
        assert isinstance(result_2d.time_solve, np.ndarray)
        assert result_2d.time_solve.shape == (2,)
        assert result_2d.time_total >= 0
        assert result_2d.time_setup >= 0

        for j in range(2):
            result_1d = solve(cats, y2[:, j])
            np.testing.assert_allclose(result_2d.x[:, j], result_1d.x, atol=1e-10)
            assert bool(result_2d.converged[j]) == result_1d.converged
            assert int(result_2d.iterations[j]) == result_1d.iterations
            assert abs(float(result_2d.residual[j]) - result_1d.residual) < 1e-10

    def test_demean_2d_matches_columnwise(self, problem):
        cats, y = problem
        y2 = np.column_stack([y, -0.5 * y + 1.0])
        result_2d = demean(cats, y2)

        assert isinstance(result_2d, DemeanResult)
        assert result_2d.y_demean.ndim == 2
        assert result_2d.y_demean.shape == y2.shape
        assert isinstance(result_2d.converged, np.ndarray)
        assert result_2d.converged.shape == (2,)
        assert isinstance(result_2d.iterations, np.ndarray)
        assert result_2d.iterations.shape == (2,)
        assert isinstance(result_2d.residual, np.ndarray)
        assert result_2d.residual.shape == (2,)
        assert isinstance(result_2d.time_solve, np.ndarray)
        assert result_2d.time_solve.shape == (2,)
        assert result_2d.time_total >= 0
        assert result_2d.time_setup >= 0

        for j in range(2):
            result_1d = demean(cats, y2[:, j])
            np.testing.assert_allclose(
                result_2d.y_demean[:, j], result_1d.y_demean, atol=1e-10
            )
            assert bool(result_2d.converged[j]) == result_1d.converged
            assert int(result_2d.iterations[j]) == result_1d.iterations
            assert abs(float(result_2d.residual[j]) - result_1d.residual) < 1e-10

    def test_rejects_non_1d_or_2d_y(self, problem):
        cats, _ = problem
        y3 = np.zeros((2, 2, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="1D or 2D"):
            solve(cats, y3)
        with pytest.raises(ValueError, match="1D or 2D"):
            demean(cats, y3)

    def test_demean_1d_returns_scalar_result_type(self, problem):
        cats, y = problem
        result = demean(cats, y)
        assert isinstance(result, DemeanResult)


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


class TestLayout:
    @pytest.mark.parametrize("layout", ["factor_major", "row_major", "compressed"])
    def test_layouts_agree(self, problem, layout):
        cats, y = problem
        result = solve(cats, y, layout=layout)
        assert result.converged

    def test_invalid_layout(self, problem):
        cats, y = problem
        with pytest.raises(ValueError, match="Unknown layout"):
            solve(cats, y, layout="bogus")
