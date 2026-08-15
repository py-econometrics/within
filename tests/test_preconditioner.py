from __future__ import annotations

import pickle

import numpy as np
import pytest

from within import LsmrOptions, Preconditioner, PreconditionerConfig, Solver, solve
from within._within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    ApproxSchurConfig,
    LocalSolverConfig,
    ReductionStrategy,
    Schur,
)

from conftest import as_solver_categories


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
    precond = solver.preconditioner
    return solver, precond, categories, y


class TestAdvancedConfigs:
    def test_preconditioner_config_additive_factory(self):
        implicit = PreconditionerConfig.additive()
        explicit = PreconditionerConfig.additive(
            local_solver=LocalSolverConfig(),
            reduction=ReductionStrategy.Auto,
        )

        assert implicit == explicit
        assert implicit == PreconditionerConfig.Additive
        assert implicit != PreconditionerConfig.Diagonal

    def test_approx_chol_config_defaults(self):
        cfg = ApproxCholConfig()
        assert cfg.seed == 0
        assert cfg.split_merge is None

    def test_approx_chol_config_custom(self):
        cfg = ApproxCholConfig(seed=42, split_merge=2)
        assert cfg.seed == 42
        assert cfg.split_merge == 2

    def test_approx_schur_config_defaults(self):
        cfg = ApproxSchurConfig()
        assert cfg.seed == 0
        assert cfg.split == 1

    def test_schur_complement_defaults(self):
        sc = LocalSolverConfig()
        assert sc.dense_threshold == 24
        # Omitted schur means the library default (approximate), not exact.
        assert sc.schur is None

    def test_schur_mode_spellings(self):
        # Omission and Schur.approximate() are the default; Schur.exact() is the
        # explicit opt-in — all three build and solve.
        rng = np.random.default_rng(7)
        cats = as_solver_categories(
            [rng.integers(0, 12, size=600), rng.integers(0, 9, size=600)]
        )
        y = rng.standard_normal(600)
        for schur in (None, Schur.approximate(), Schur.exact()):
            local = LocalSolverConfig(schur=schur)
            assert local.schur is schur
            result = solve(
                cats,
                y,
                options=LsmrOptions(maxiter=2000),
                preconditioner=AdditiveSchwarz(local_solver=local),
            )
            assert result.converged

    def test_schur_complement_solve(self, problem):
        cats, y = problem
        result = solve(
            as_solver_categories(cats),
            y,
            options=LsmrOptions(),
            preconditioner=AdditiveSchwarz(local_solver=LocalSolverConfig()),
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

    def test_preconditioner_nrows_ncols_match_solver(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        assert precond.nrows == precond.ncols == solver.n_dofs

    def test_preconditioner_constructor_roundtrip(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        data = pickle.dumps(precond)
        precond2 = pickle.loads(data)
        x = np.random.randn(precond.nrows)
        np.testing.assert_array_equal(precond.apply(x), precond2.apply(x))
        assert precond2.config == precond.config

    def test_preconditioner_exposes_default_config(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        assert precond.config == PreconditionerConfig.Additive

    def test_preconditioner_exposes_tuned_config(self, problem):
        cats, _ = problem
        requested = PreconditionerConfig.additive(
            local_solver=LocalSolverConfig(
                approx_chol=ApproxCholConfig(seed=41, split_merge=3),
                schur=Schur.approximate(ApproxSchurConfig(seed=17, split=2)),
                dense_threshold=0,
            ),
            reduction=ReductionStrategy.AtomicScatter,
        )

        solver = Solver(as_solver_categories(cats), preconditioner=requested)
        precond = solver.preconditioner

        assert precond is not None
        assert precond.config == requested

    def test_preconditioner_corrupt_bytes_raises(self):
        with pytest.raises(ValueError):
            Preconditioner(b"garbage")

    def test_preconditioner_apply_deterministic(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        x = np.random.randn(precond.nrows)
        r1 = precond.apply(x)
        r2 = precond.apply(x)
        np.testing.assert_array_equal(r1, r2)

    def test_preconditioner_repr_diagonal(self):
        categories = as_solver_categories(
            [np.array([0, 1, 0, 1, 2, 2]), np.array([0, 0, 1, 1, 0, 1])]
        )
        solver = Solver(categories, preconditioner=PreconditionerConfig.Diagonal)
        precond = solver.preconditioner

        assert precond is not None
        assert "Diagonal" in repr(precond)
        assert precond.config == PreconditionerConfig.Diagonal

    def test_single_factor_diagonal_preconditioner_is_cached(self):
        categories = np.asfortranarray(
            np.array([[0], [1], [0], [2], [1]], dtype=np.uint32)
        )
        solver = Solver(categories, preconditioner=PreconditionerConfig.Diagonal)
        precond = solver.preconditioner

        assert precond is not None
        assert precond.nrows == precond.ncols == 3
