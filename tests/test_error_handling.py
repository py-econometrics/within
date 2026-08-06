from __future__ import annotations

import numpy as np
import pytest

from within import Effect, Solver, solve, solve_batch
from within._within import ApproxSchurConfig
from within.config import AdditiveSchwarz

from conftest import as_solver_categories


class TestErrorHandling:
    def test_empty_categories_raises(self):
        """0-row categories should raise ValueError."""
        cats = np.empty((0, 2), dtype=np.uint32, order="F")
        y = np.array([], dtype=np.float64)
        with pytest.raises(ValueError):
            solve(cats, y)

    def test_mismatched_y_length_raises(self):
        """len(y) != n_obs should raise."""
        cats = as_solver_categories([np.array([0, 1, 0]), np.array([0, 0, 1])])
        y = np.array([1.0, 2.0])  # wrong length
        with pytest.raises((ValueError, BaseException)):
            solve(cats, y)

    def test_mismatched_weights_length_raises(self):
        """len(weights) != n_obs should raise."""
        cats = as_solver_categories([np.array([0, 1, 0]), np.array([0, 0, 1])])
        y = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 2.0])  # wrong length
        with pytest.raises(ValueError):
            solve(cats, y, weights=weights)

    def test_wrong_dtype_categories(self):
        """float64 categories should raise a TypeError naming uint32."""
        cats = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64, order="F")
        y = np.array([1.0, 2.0])
        with pytest.raises(TypeError, match="uint32"):
            solve(cats, y)

    def test_int64_categories_raises_typeerror(self):
        """int64 categories (the pandas.factorize default) should raise a TypeError naming uint32."""
        cats = np.array([[0, 0], [1, 1]], dtype=np.int64, order="F")
        y = np.array([1.0, 2.0])
        with pytest.raises(TypeError, match="uint32"):
            solve(cats, y)

    def test_wrong_dtype_y(self):
        """int32 y should raise a TypeError naming float64."""
        cats = as_solver_categories([np.array([0, 1, 0]), np.array([0, 0, 1])])
        y = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(TypeError, match="float64"):
            solve(cats, y)

    def test_wrong_dtype_weights(self):
        """float32 weights should raise a TypeError naming float64."""
        cats = as_solver_categories([np.array([0, 1, 0]), np.array([0, 0, 1])])
        y = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        with pytest.raises(TypeError, match="float64"):
            solve(cats, y, weights=w)

    def test_1d_categories_raises(self):
        """1-D categories should raise TypeError."""
        cats = np.array([0, 1, 2], dtype=np.uint32)
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(TypeError):
            solve(cats, y)

    def test_invalid_config_type_raises(self):
        """String config should raise TypeError."""
        cats = as_solver_categories([np.array([0, 1, 0]), np.array([0, 0, 1])])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(TypeError):
            solve(cats, y, options="invalid")

    def test_invalid_preconditioner_type_raises(self):
        """String preconditioner should raise TypeError."""
        cats = as_solver_categories([np.array([0, 1, 0]), np.array([0, 0, 1])])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(TypeError) as exc:
            solve(cats, y, preconditioner="invalid")
        assert "PreconditionerConfig.Diagonal" in str(exc.value)

    @pytest.mark.parametrize(
        "call",
        [
            lambda design: solve(design, np.array([1.0, 2.0, 3.0])),
            lambda design: solve_batch(design, np.ones((3, 2))),
            lambda design: Solver(design),
        ],
        ids=["solve", "solve_batch", "Solver"],
    )
    def test_invalid_design_type_raises(self, call):
        """A design that is neither array nor list of Effect should raise TypeError."""
        with pytest.raises(TypeError, match="2-D uint32 array or a list of Effect"):
            call("invalid")

    def test_additive_schwarz_rejects_local_solver_type(self):
        """A wrong-type local_solver should raise at construction, not at solve."""
        with pytest.raises(TypeError):
            AdditiveSchwarz(local_solver="invalid")

    def test_approx_schur_config_split_zero_raises(self):
        """ApproxSchurConfig(split=0) should raise ValueError."""
        with pytest.raises((ValueError, OverflowError)):
            ApproxSchurConfig(split=0)


class TestEffectErrors:
    def test_empty_effect_raises(self):
        with pytest.raises(ValueError, match="intercept or at least one slope"):
            Effect(np.array([0, 1, 0], dtype=np.uint32), intercept=False)

    def test_slope_length_mismatch_names_slope(self):
        levels = np.array([0, 1, 0], dtype=np.uint32)
        with pytest.raises(ValueError, match="slope 0"):
            Effect(levels, intercept=True, slopes=[np.array([1.0, 2.0])])

    def test_slope_term_alongside_other_terms_solves(self):
        levels = np.array([0, 1, 0], dtype=np.uint32)
        slope = [np.array([1.0, 2.0, 3.0])]
        y = np.array([1.0, 2.0, 3.0])
        # Cross-factor routing (#61): slope terms solve alongside others.
        result = solve(
            [
                Effect(levels, intercept=True, slopes=slope),
                Effect(levels, intercept=True),
            ],
            y,
        )
        assert result.converged
        assert np.all(np.isfinite(result.x))
