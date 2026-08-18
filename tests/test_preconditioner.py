from __future__ import annotations

import pickle

import numpy as np
import pytest
from conftest import as_solver_categories
from within import LsmrOptions, Preconditioner, PreconditionerConfig, Solver, solve
from within._within import (
    ApproxCholConfig,
    ApproxSchurConfig,
    LocalSolverConfig,
    ReductionStrategy,
    Schur,
)


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
    def test_preconditioner_config_additive_constructor(self):
        implicit = PreconditionerConfig.Additive()
        explicit = PreconditionerConfig.Additive(
            local_solver=LocalSolverConfig(),
            reduction=ReductionStrategy.Auto,
        )

        assert implicit == explicit
        assert implicit != PreconditionerConfig.Diagonal()
        assert implicit.local_solver == LocalSolverConfig()

        with pytest.raises(TypeError):
            PreconditionerConfig.Additive(local_solver=None)

    def test_preconditioner_config_variant_introspection(self):
        # Each variant is a subclass of PreconditionerConfig (the 3.9-safe
        # equivalent of ``match``), and Additive exposes its fields as getters.
        off = PreconditionerConfig.Off()
        tuned = PreconditionerConfig.Additive(
            local_solver=LocalSolverConfig(dense_threshold=0),
            reduction=ReductionStrategy.AtomicScatter,
        )

        assert isinstance(off, PreconditionerConfig)
        assert isinstance(tuned, PreconditionerConfig.Additive)
        assert not isinstance(off, PreconditionerConfig.Additive)
        assert tuned.reduction == ReductionStrategy.AtomicScatter
        assert tuned.local_solver.dense_threshold == 0

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
        assert sc.approx_chol.seed == 0
        assert sc.approx_chol.split_merge == 2
        assert repr(sc.schur) == "Schur.approximate(seed=0, split=1)"
        assert sc.scaling.on_failure == "warn"

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
            expected = Schur.approximate() if schur is None else schur
            assert repr(local.schur) == repr(expected)
            result = solve(
                cats,
                y,
                options=LsmrOptions(maxiter=2000),
                preconditioner=PreconditionerConfig.Additive(local_solver=local),
            )
            assert result.converged

    def test_schur_complement_solve(self, problem):
        cats, y = problem
        result = solve(
            as_solver_categories(cats),
            y,
            options=LsmrOptions(),
            preconditioner=PreconditionerConfig.Additive(
                local_solver=LocalSolverConfig()
            ),
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

    def test_preconditioner_exposes_build_time(self):
        categories = as_solver_categories(
            [np.array([0, 1, 0, 1, 2, 2]), np.array([0, 0, 1, 1, 0, 1])]
        )

        for config in (
            PreconditionerConfig.Additive(),
            PreconditionerConfig.Diagonal(),
        ):
            precond = Solver(categories, preconditioner=config).preconditioner
            assert precond is not None
            assert isinstance(precond.build_time_seconds, float)
            assert np.isfinite(precond.build_time_seconds)
            assert precond.build_time_seconds >= 0.0

    def test_preconditioner_constructor_roundtrip(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        data = pickle.dumps(precond)
        precond2 = pickle.loads(data)
        x = np.random.randn(precond.nrows)
        np.testing.assert_array_equal(precond.apply(x), precond2.apply(x))
        assert precond2.config == precond.config
        assert precond2.build_time_seconds == precond.build_time_seconds

    def test_preconditioner_exposes_default_config(self, solver_and_precond):
        solver, precond, categories, y = solver_and_precond
        assert precond.config == PreconditionerConfig.Additive()

    def test_preconditioner_exposes_tuned_config(self, problem):
        cats, _ = problem
        requested = PreconditionerConfig.Additive(
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
        actual = precond.config
        assert actual == requested
        assert isinstance(actual, PreconditionerConfig.Additive)
        assert actual.reduction == ReductionStrategy.AtomicScatter
        assert actual.local_solver.approx_chol.seed == 41
        assert actual.local_solver.approx_chol.split_merge == 3
        assert repr(actual.local_solver.schur) == "Schur.approximate(seed=17, split=2)"
        assert actual.local_solver.dense_threshold == 0
        assert actual.local_solver.scaling.on_failure == "warn"

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
        solver = Solver(categories, preconditioner=PreconditionerConfig.Diagonal())
        precond = solver.preconditioner

        assert precond is not None
        assert "Diagonal" in repr(precond)
        assert precond.config == PreconditionerConfig.Diagonal()

    def test_single_factor_diagonal_preconditioner_is_cached(self):
        categories = np.asfortranarray(
            np.array([[0], [1], [0], [2], [1]], dtype=np.uint32)
        )
        solver = Solver(categories, preconditioner=PreconditionerConfig.Diagonal())
        precond = solver.preconditioner

        assert precond is not None
        assert precond.nrows == precond.ncols == 3
