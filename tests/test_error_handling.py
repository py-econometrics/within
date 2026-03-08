from __future__ import annotations

import numpy as np
import pytest

from within import solve
from within._within import ApproxCholConfig, ApproxSchurConfig


def as_solver_categories(cats):
    return np.asfortranarray(np.column_stack(cats).astype(np.uint32))


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
        """float64 categories should raise TypeError."""
        cats = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64, order="F")
        y = np.array([1.0, 2.0])
        with pytest.raises(TypeError):
            solve(cats, y)

    def test_wrong_dtype_y(self):
        """int32 y should raise TypeError."""
        cats = as_solver_categories([np.array([0, 1, 0]), np.array([0, 0, 1])])
        y = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(TypeError):
            solve(cats, y)

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
            solve(cats, y, config="invalid")

    def test_invalid_preconditioner_type_raises(self):
        """String preconditioner should raise TypeError."""
        cats = as_solver_categories([np.array([0, 1, 0]), np.array([0, 0, 1])])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(TypeError):
            solve(cats, y, preconditioner="invalid")

    def test_approx_chol_config_split_zero_raises(self):
        """ApproxCholConfig(split=0) should raise ValueError."""
        with pytest.raises((ValueError, OverflowError)):
            ApproxCholConfig(split=0)

    def test_approx_schur_config_split_zero_raises(self):
        """ApproxSchurConfig(split=0) should raise ValueError."""
        with pytest.raises((ValueError, OverflowError)):
            ApproxSchurConfig(split=0)
