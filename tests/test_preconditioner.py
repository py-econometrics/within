from __future__ import annotations

import pickle

import numpy as np
import pytest

from within import CG, GMRES, FePreconditioner, Preconditioner, Solver, solve
from within._within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    ApproxSchurConfig,
    FullSddm,
    SchurComplement,
)


def as_solver_categories(cats):
    return np.asfortranarray(np.column_stack(cats).astype(np.uint32))


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


@pytest.fixture()
def solver_and_precond(problem):
    """Build a Solver and extract its preconditioner."""
    cats, y = problem
    categories = as_solver_categories(cats)
    solver = Solver(categories)
    precond = solver.preconditioner()
    return solver, precond, categories, y


class TestAdvancedConfigs:
    def test_approx_chol_config_defaults(self):
        cfg = ApproxCholConfig()
        assert cfg.seed == 0
        assert cfg.split == 1

    def test_approx_chol_config_custom(self):
        cfg = ApproxCholConfig(seed=42, split=2)
        assert cfg.seed == 42
        assert cfg.split == 2

    def test_approx_schur_config_defaults(self):
        cfg = ApproxSchurConfig()
        assert cfg.seed == 0
        assert cfg.split == 1

    def test_schur_complement_defaults(self):
        sc = SchurComplement()
        assert sc.dense_threshold == 24

    def test_schur_complement_solve(self, problem):
        cats, y = problem
        result = solve(
            as_solver_categories(cats),
            y,
            CG(),
            preconditioner=AdditiveSchwarz(local_solver=SchurComplement()),
        )
        assert result.converged

    def test_full_sddm_defaults(self):
        fs = FullSddm()
        assert fs.approx_chol is None

    def test_full_sddm_solve(self, problem):
        cats, y = problem
        result = solve(
            as_solver_categories(cats),
            y,
            CG(),
            preconditioner=AdditiveSchwarz(local_solver=FullSddm()),
        )
        assert result.converged


class TestFePreconditioner:
    def test_preconditioner_apply(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        x = np.random.randn(precond.nrows)
        result = precond.apply(x)
        assert result.shape == (precond.nrows,)
        assert np.all(np.isfinite(result))

    def test_preconditioner_apply_wrong_length_raises(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        x = np.random.randn(precond.nrows + 5)
        with pytest.raises(ValueError):
            precond.apply(x)

    def test_preconditioner_repr_additive(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        assert "Additive" in repr(precond)

    def test_preconditioner_repr_multiplicative(self, problem):
        cats, y = problem
        solver = Solver(
            as_solver_categories(cats),
            GMRES(),
            preconditioner=Preconditioner.Multiplicative,
        )
        precond = solver.preconditioner()
        assert "Multiplicative" in repr(precond)

    def test_preconditioner_nrows_ncols_match_solver(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        assert precond.nrows == precond.ncols == solver.n_dofs

    def test_preconditioner_constructor_roundtrip(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        data = pickle.dumps(precond)
        precond2 = pickle.loads(data)
        x = np.random.randn(precond.nrows)
        np.testing.assert_array_equal(precond.apply(x), precond2.apply(x))

    def test_preconditioner_corrupt_bytes_raises(self):
        with pytest.raises(ValueError):
            FePreconditioner(b"garbage")

    def test_preconditioner_apply_deterministic(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        x = np.random.randn(precond.nrows)
        r1 = precond.apply(x)
        r2 = precond.apply(x)
        np.testing.assert_array_equal(r1, r2)
