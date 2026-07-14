from __future__ import annotations

import warnings

import numpy as np
import pytest

from within import (
    BatchSolveResult,
    CoefficientLayout,
    Effect,
    LsmrOptions,
    Preconditioner,
    PreconditionerConfig,
    Solver,
    solve,
)
from within.config import AdditiveSchwarz, LocalSolverConfig, ScalingConfig

from conftest import generate_synthetic_data


def as_solver_categories(cats):
    return np.asfortranarray(np.column_stack(cats).astype(np.uint32))


def test_coefficient_layout_locates_unidentified_and_round_trips():
    # firm level 2 is a singleton in the (non-first) slope term, so its slope
    # direction is unidentified.
    worker = np.array([0, 0, 1, 1, 2, 2], np.uint32)
    firm = np.array([2, 0, 0, 1, 1, 0], np.uint32)
    x = np.array([0.5, -1.0, 0.3, 2.0, -0.7, 1.1])
    y = np.array([1.0, 2.0, 3.0, 1.5, 0.4, -0.2])

    res = solve([Effect(worker, True), Effect(firm, True, [x])], y)
    layout = res.layout
    assert isinstance(layout, CoefficientLayout)
    assert layout.n_dofs() == len(res.x)
    assert [
        (layout.n_levels(t), layout.n_columns(t)) for t in range(layout.n_terms())
    ] == [
        (3, 1),
        (3, 2),
    ]

    # The reported unidentified direction resolves to a zero coefficient slot.
    (u,) = res.unidentified
    assert res.x[layout.index(u.term, u.level, u.column)] == 0.0

    # index and address are mutual inverses over the whole vector.
    for i in range(layout.n_dofs()):
        assert layout.index(*layout.address(i)) == i

    # Out-of-range coordinates raise instead of reading the wrong coefficient.
    with pytest.raises(IndexError):
        layout.index(u.term, u.level, layout.n_columns(u.term))
    with pytest.raises(IndexError):
        layout.address(layout.n_dofs())


def test_free_solve_positional_order_is_weights_then_options():
    # Pins (design, y, weights, options, ...): a weights array 3rd and
    # LsmrOptions 4th must be accepted. A re-swap would parse the array as
    # options (and LsmrOptions as weights) and raise.
    rng = np.random.default_rng(0)
    cats = as_solver_categories(
        [rng.integers(0, 8, size=300), rng.integers(0, 6, size=300)]
    )
    y = rng.standard_normal(300)
    w = rng.uniform(0.5, 2.0, size=300)
    result = solve(cats, y, w, LsmrOptions(maxiter=2000))
    assert result.converged


def test_one_shot_solve_surfaces_build_warnings():
    # A frustrated slope design with scaling certification disabled emits a
    # build warning; the one-shot solve/solve_batch must re-emit it, matching
    # the persistent Solver (#103).
    f = np.array([0, 0, 0, 1, 1, 1], np.uint32)
    g = np.array([0, 1, 2, 0, 1, 2], np.uint32)
    z = np.array([-2.0, 1.0, 1.0, -1.0, -1.0, 2.0])
    y = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.5])
    from within import solve_batch

    design = [Effect(f, True, [z]), Effect(g, True)]
    precond = AdditiveSchwarz(
        local_solver=LocalSolverConfig(scaling=ScalingConfig(max_sweeps=0))
    )

    def user_warnings(call):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            call()
        return [str(w.message) for w in record if issubclass(w.category, UserWarning)]

    persistent = user_warnings(lambda: Solver(design, preconditioner=precond))
    assert persistent, "fixture should emit a build warning"
    assert user_warnings(lambda: solve(design, y, preconditioner=precond)) == persistent
    assert (
        user_warnings(
            lambda: solve_batch(design, np.column_stack([y, y]), preconditioner=precond)
        )
        == persistent
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


class TestSolveDefaults:
    def test_default_config(self, problem):
        cats, y = problem
        result = solve(as_solver_categories(cats), y)
        assert result.converged
        assert result.iterations > 0
        assert result.residual < 1e-6

    def test_unpreconditioned(self, problem):
        cats, y = problem
        result = solve(
            as_solver_categories(cats),
            y,
            options=LsmrOptions(),
            preconditioner=PreconditionerConfig.Off,
        )
        assert result.converged


class TestSolveWeighted:
    def test_weighted(self, problem):
        cats, y = problem
        weights = np.random.exponential(1.0, size=len(y))
        result = solve(as_solver_categories(cats), y, weights=weights)
        assert result.converged


class TestPreconditioners:
    def test_additive_schwarz(self, problem):
        cats, y = problem
        result = solve(
            as_solver_categories(cats),
            y,
            options=LsmrOptions(),
            preconditioner=PreconditionerConfig.Additive,
        )
        assert result.converged

    def test_advanced_additive_schwarz(self, problem):
        """Test advanced config via AdditiveSchwarz."""
        cats, y = problem
        result = solve(
            as_solver_categories(cats),
            y,
            options=LsmrOptions(),
            preconditioner=AdditiveSchwarz(),
        )
        assert result.converged

    def test_diagonal_preconditioner(self):
        rng = np.random.default_rng(123)
        categories = as_solver_categories(
            [rng.integers(0, 10, size=400), rng.integers(0, 8, size=400)]
        )
        y = rng.standard_normal(400)
        result = solve(
            categories,
            y,
            options=LsmrOptions(maxiter=2000),
            preconditioner=PreconditionerConfig.Diagonal,
        )
        assert result.converged
        assert np.all(np.isfinite(result.x))


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

    def test_demeaned_field(self, problem):
        """The demeaned field should equal y - D*x."""
        cats, y = problem
        result = solve(as_solver_categories(cats), y)
        assert result.demeaned.shape == y.shape
        # Recompute y - D*x manually
        expected = y.copy()
        offset = 0
        for f in range(2):
            for lvl in range(50):
                expected[cats[f] == lvl] -= result.x[offset + lvl]
            offset += 50
        np.testing.assert_allclose(result.demeaned, expected, atol=1e-10)


class TestSolveResult:
    def test_result_fields(self, problem):
        cats, y = problem
        result = solve(as_solver_categories(cats), y)
        assert isinstance(result.x, np.ndarray)
        assert result.x.dtype == np.float64
        assert len(result.x) == 100  # 50 + 50 levels
        assert result.unidentified == []
        assert result.time_total >= 0
        assert result.time_setup >= 0
        assert result.time_solve >= 0


class TestContiguityWarning:
    def test_c_contiguous_warns(self, problem):
        cats, y = problem
        cats_c = np.column_stack(cats).astype(np.uint32)  # C-contiguous
        assert not cats_c.flags["F_CONTIGUOUS"]
        with pytest.warns(UserWarning, match="not F-contiguous"):
            solve(cats_c, y)

    def test_f_contiguous_no_warning(self, problem):
        cats, y = problem
        cats_f = as_solver_categories(cats)  # F-contiguous
        assert cats_f.flags["F_CONTIGUOUS"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solve(cats_f, y)

    def test_column_reversed_f_view_no_warning(self, problem):
        # Reversing columns keeps each column contiguous (row stride == itemsize),
        # so ingest borrows them zero-copy and no warning applies.
        cats, y = problem
        cats_rev = as_solver_categories(cats)[:, ::-1]
        assert cats_rev.strides[0] == cats_rev.itemsize
        assert cats_rev.strides[1] < 0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solve(cats_rev, y)


# ---------------------------------------------------------------------------
# Solver class tests
# ---------------------------------------------------------------------------


class TestSolver:
    def test_solver_matches_oneshot(self, problem):
        """Solver.solve() matches one-shot solve()."""
        cats, y = problem
        categories = as_solver_categories(cats)
        oneshot = solve(categories, y)

        solver = Solver(categories)
        result = solver.solve(y)
        assert result.converged
        np.testing.assert_allclose(result.x, oneshot.x, atol=1e-10)
        np.testing.assert_allclose(result.demeaned, oneshot.demeaned, atol=1e-10)

    def test_solver_reuse(self, problem):
        """Multiple solves with same Solver give consistent results."""
        cats, y = problem
        categories = as_solver_categories(cats)
        solver = Solver(categories)
        r1 = solver.solve(y)
        r2 = solver.solve(y)
        np.testing.assert_array_equal(r1.x, r2.x)

    def test_solver_no_preconditioner(self, problem):
        """Solver with PreconditionerConfig.Off works."""
        cats, y = problem
        solver = Solver(
            as_solver_categories(cats), preconditioner=PreconditionerConfig.Off
        )
        result = solver.solve(y)
        assert result.converged

    def test_solver_properties(self, problem):
        cats, y = problem
        solver = Solver(as_solver_categories(cats))
        assert solver.n_dofs == 100
        assert solver.n_obs == 10_000


class TestSolverBatch:
    def test_batch_matches_individual(self, problem):
        """Batch solve matches individual solves."""
        cats, y = problem
        categories = as_solver_categories(cats)
        solver = Solver(categories)

        y2 = np.random.randn(10_000)
        Y = np.column_stack([y, y2])

        batch = solver.solve_batch(Y)
        r1 = solver.solve(y)
        r2 = solver.solve(y2)

        assert isinstance(batch, BatchSolveResult)
        assert batch.x.shape == (100, 2)
        assert batch.demeaned.shape == (10_000, 2)
        np.testing.assert_allclose(batch.x[:, 0], r1.x, atol=1e-10)
        np.testing.assert_allclose(batch.x[:, 1], r2.x, atol=1e-10)
        np.testing.assert_allclose(batch.demeaned[:, 0], r1.demeaned, atol=1e-10)
        np.testing.assert_allclose(batch.demeaned[:, 1], r2.demeaned, atol=1e-10)

    def test_batch_result_fields(self, problem):
        cats, y = problem
        solver = Solver(as_solver_categories(cats))
        Y = np.column_stack([y, y])
        batch = solver.solve_batch(Y)
        assert len(batch.converged) == 2
        assert len(batch.iterations) == 2
        assert len(batch.residual) == 2
        assert len(batch.time_solve) == 2
        assert batch.time_total >= 0
        assert batch.unidentified == []


class TestSolverSerde:
    def test_preconditioner_roundtrip(self, problem):
        """Extract preconditioner object, reuse in new solver."""
        cats, y = problem
        categories = as_solver_categories(cats)

        solver1 = Solver(categories)
        r1 = solver1.solve(y)

        precond = solver1.preconditioner
        assert isinstance(precond, Preconditioner)
        assert precond.nrows > 0

        # Reuse in new solver
        solver2 = Solver(categories, preconditioner=precond)
        r2 = solver2.solve(y)
        np.testing.assert_allclose(r2.x, r1.x, atol=1e-10)

    def test_preconditioner_pickle(self, problem):
        """Pickle roundtrip of Preconditioner."""
        import pickle

        cats, y = problem
        categories = as_solver_categories(cats)

        solver1 = Solver(categories)
        r1 = solver1.solve(y)

        precond = solver1.preconditioner
        data = pickle.dumps(precond)
        precond2 = pickle.loads(data)

        solver2 = Solver(categories, preconditioner=precond2)
        r2 = solver2.solve(y)
        np.testing.assert_allclose(r2.x, r1.x, atol=1e-10)

    def test_no_preconditioner_returns_none(self, problem):
        cats, y = problem
        solver = Solver(
            as_solver_categories(cats), preconditioner=PreconditionerConfig.Off
        )
        assert solver.preconditioner is None

    def test_diagonal_preconditioner_pickle_and_reuse(self):
        import pickle

        categories = as_solver_categories(
            [np.array([0, 1, 0, 1, 2, 2]), np.array([0, 0, 1, 1, 0, 1])]
        )
        y = np.array([1.0, 2.0, 1.5, 2.5, 3.0, 3.5])

        solver1 = Solver(categories, preconditioner=PreconditionerConfig.Diagonal)
        r1 = solver1.solve(y)
        precond = solver1.preconditioner

        assert precond is not None
        assert "Diagonal" in repr(precond)

        x = np.arange(precond.ncols, dtype=np.float64) + 0.25
        np.testing.assert_array_equal(precond.apply(x), precond.apply(x))

        precond2 = pickle.loads(pickle.dumps(precond))
        np.testing.assert_array_equal(precond.apply(x), precond2.apply(x))

        solver2 = Solver(categories, preconditioner=precond2)
        r2 = solver2.solve(y)
        np.testing.assert_allclose(r2.x, r1.x, atol=1e-10)


# ---------------------------------------------------------------------------
# Convenience alias tests
# ---------------------------------------------------------------------------


class TestAliases:
    def test_additive_alias(self, problem):
        cats, y = problem
        result = solve(as_solver_categories(cats), y, preconditioner=AdditiveSchwarz())
        assert result.converged


class TestSolveBatchFreeFunction:
    """Tests for the free-function solve_batch()."""

    def test_solve_batch_basic(self, problem):
        cats, y = problem
        categories = as_solver_categories(cats)
        Y = np.column_stack([y, np.random.randn(len(y))])
        from within import solve_batch

        result = solve_batch(categories, Y)
        assert all(result.converged)

    def test_solve_batch_matches_individual(self, problem):
        cats, y = problem
        categories = as_solver_categories(cats)
        y2 = np.random.randn(len(y))
        Y = np.column_stack([y, y2])
        from within import solve_batch

        batch = solve_batch(categories, Y)
        r1 = solve(categories, y)
        r2 = solve(categories, y2)
        np.testing.assert_allclose(batch.x[:, 0], r1.x, atol=1e-10)
        np.testing.assert_allclose(batch.x[:, 1], r2.x, atol=1e-10)

    def test_solve_batch_effect_terms_match_individual(self):
        f = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32)
        g = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32)
        z = np.array([-2.0, 1.0, 1.0, -1.0, -1.0, 2.0])
        Y = np.column_stack(
            [
                np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.5]),
                np.array([0.3, 1.1, -0.7, 2.2, 0.9, -1.4]),
            ]
        )
        from within import Effect, solve_batch

        effects = [Effect(f, True, [z]), Effect(g, True)]
        batch = solve_batch(effects, Y)
        for j in range(Y.shape[1]):
            single = solve(effects, Y[:, j].copy())
            np.testing.assert_allclose(batch.x[:, j], single.x, atol=1e-10)
            np.testing.assert_allclose(
                batch.demeaned[:, j], single.demeaned, atol=1e-10
            )

    def test_solve_batch_result_shapes(self, problem):
        cats, y = problem
        categories = as_solver_categories(cats)
        k = 3
        Y = np.column_stack([np.random.randn(len(y)) for _ in range(k)])
        from within import solve_batch

        result = solve_batch(categories, Y)
        assert result.x.shape == (100, k)  # 50+50 dofs, k columns
        assert result.demeaned.shape == (len(y), k)

    def test_solve_batch_weighted(self, problem):
        cats, y = problem
        categories = as_solver_categories(cats)
        weights = np.random.exponential(1.0, size=len(y))
        Y = np.column_stack([y, np.random.randn(len(y))])
        from within import solve_batch

        result = solve_batch(categories, Y, weights=weights)
        assert all(result.converged)

    def test_solve_batch_single_column(self, problem):
        cats, y = problem
        categories = as_solver_categories(cats)
        Y = y.reshape(-1, 1)
        from within import solve_batch

        result = solve_batch(categories, Y)
        assert result.x.shape[1] == 1
        assert all(result.converged)


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


class TestEffectDesign:
    def test_intercept_only_matches_categories_path(self):
        # The issue's trust anchor: an intercept-only effect design must be
        # bit-identical to the equivalent categories-matrix solve.
        rng = np.random.default_rng(0)
        n = 300
        col0 = rng.integers(0, 4, size=n).astype(np.uint32)
        col1 = rng.integers(0, 6, size=n).astype(np.uint32)
        y = rng.standard_normal(n)

        categories = np.asfortranarray(np.column_stack([col0, col1]))
        from_categories = solve(categories, y)
        from_effects = solve(
            [Effect(col0, intercept=True), Effect(col1, intercept=True)], y
        )

        assert from_effects.converged
        np.testing.assert_array_equal(from_effects.x, from_categories.x)
        np.testing.assert_array_equal(from_effects.demeaned, from_categories.demeaned)
