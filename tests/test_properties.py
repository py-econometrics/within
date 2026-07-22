from __future__ import annotations

import pickle

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from within import Solver, solve
from within.config import AdditiveSchwarz, ReductionStrategy


@st.composite
def random_fe_problem(draw):
    """Generate a random fixed-effects problem.

    Returns (categories, y) where categories is F-contiguous uint32.
    """
    n_factors = draw(st.integers(min_value=2, max_value=3))
    n_levels = [draw(st.integers(min_value=2, max_value=50)) for _ in range(n_factors)]
    n_obs = draw(st.integers(min_value=50, max_value=2000))

    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))
    cats = [rng.integers(0, nl, size=n_obs, dtype=np.uint32) for nl in n_levels]
    categories = np.asfortranarray(np.column_stack(cats))
    y = rng.standard_normal(n_obs)
    return categories, y


class TestProperties:
    # Mathematical properties (residual orthogonality, demeaned == y - D*x,
    # solve() == Solver().solve()) are proptested with stronger, unguarded
    # oracles on the Rust side (properties.rs, property_gaps.rs). Kept here:
    # determinism through the binding, and pickle roundtrip (Python-only path).

    @given(data=random_fe_problem())
    @settings(
        max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_solver_determinism(self, data):
        """Same input solved twice gives identical result."""
        categories, y = data
        r1 = solve(categories, y)
        r2 = solve(categories, y)
        np.testing.assert_allclose(r1.x, r2.x, atol=1e-14)

    @given(data=random_fe_problem())
    @settings(
        max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_preconditioner_pickle_preserves_apply(self, data):
        """Pickle roundtrip of preconditioner preserves apply()."""
        categories, y = data
        solver = Solver(categories)
        precond = solver.preconditioner
        if precond is None:
            return

        x = np.random.randn(precond.nrows)
        result_before = precond.apply(x)

        data_bytes = pickle.dumps(precond)
        precond2 = pickle.loads(data_bytes)
        result_after = precond2.apply(x)

        np.testing.assert_array_equal(result_before, result_after)


class TestAdvancedPreconditioners:
    """Tests for AdditiveSchwarz preconditioner configs."""

    def test_additive_schwarz_object_converges(self):
        """AdditiveSchwarz() as preconditioner object should converge."""
        rng = np.random.default_rng(42)
        categories = np.asfortranarray(
            np.column_stack(
                [rng.integers(0, 20, size=500), rng.integers(0, 20, size=500)]
            ).astype(np.uint32)
        )
        y = rng.standard_normal(500)
        result = solve(categories, y, preconditioner=AdditiveSchwarz())
        assert result.converged

    def test_reduction_strategy_atomic_scatter_converges(self):
        """AdditiveSchwarz with AtomicScatter reduction should converge."""
        rng = np.random.default_rng(10)
        categories = np.asfortranarray(
            np.column_stack(
                [rng.integers(0, 20, size=500), rng.integers(0, 20, size=500)]
            ).astype(np.uint32)
        )
        y = rng.standard_normal(500)
        result = solve(
            categories,
            y,
            preconditioner=AdditiveSchwarz(reduction=ReductionStrategy.AtomicScatter),
        )
        assert result.converged

    def test_reduction_strategy_parallel_reduction_converges(self):
        """AdditiveSchwarz with ParallelReduction strategy should converge."""
        rng = np.random.default_rng(11)
        categories = np.asfortranarray(
            np.column_stack(
                [rng.integers(0, 20, size=500), rng.integers(0, 20, size=500)]
            ).astype(np.uint32)
        )
        y = rng.standard_normal(500)
        result = solve(
            categories,
            y,
            preconditioner=AdditiveSchwarz(
                reduction=ReductionStrategy.ParallelReduction
            ),
        )
        assert result.converged

    def test_reduction_strategies_give_equivalent_solutions(self):
        """AtomicScatter and ParallelReduction should give the same solution."""
        rng = np.random.default_rng(42)
        categories = np.asfortranarray(
            np.column_stack(
                [rng.integers(0, 20, size=500), rng.integers(0, 20, size=500)]
            ).astype(np.uint32)
        )
        y = rng.standard_normal(500)
        r_atomic = solve(
            categories,
            y,
            preconditioner=AdditiveSchwarz(reduction=ReductionStrategy.AtomicScatter),
        )
        r_parallel = solve(
            categories,
            y,
            preconditioner=AdditiveSchwarz(
                reduction=ReductionStrategy.ParallelReduction
            ),
        )
        if r_atomic.converged and r_parallel.converged:
            np.testing.assert_allclose(r_atomic.x, r_parallel.x, atol=1e-4)


class TestBatchProperties:
    """Property tests for solve_batch / Solver.solve_batch."""

    def test_batch_identical_columns_give_identical_results(self):
        """Repeating the same column in a batch should give identical coefficient columns."""
        rng = np.random.default_rng(42)
        categories = np.asfortranarray(
            np.column_stack(
                [rng.integers(0, 20, size=500), rng.integers(0, 20, size=500)]
            ).astype(np.uint32)
        )
        y = rng.standard_normal(500)
        Y = np.column_stack([y, y, y])
        solver = Solver(categories)
        batch = solver.solve_batch(Y)
        np.testing.assert_allclose(batch.x[:, 0], batch.x[:, 1], atol=1e-12)
        np.testing.assert_allclose(batch.x[:, 1], batch.x[:, 2], atol=1e-12)

    def test_batch_single_column_matches_single_solve(self):
        """A batch with one column should match a direct solve()."""
        rng = np.random.default_rng(7)
        categories = np.asfortranarray(
            np.column_stack(
                [rng.integers(0, 15, size=400), rng.integers(0, 15, size=400)]
            ).astype(np.uint32)
        )
        y = rng.standard_normal(400)
        Y = y[:, np.newaxis]
        solver = Solver(categories)
        batch = solver.solve_batch(Y)
        single = solver.solve(y)
        if batch.converged[0] and single.converged:
            np.testing.assert_allclose(batch.x[:, 0], single.x, atol=1e-12)

    def test_batch_x_shape(self):
        """batch.x should have shape (n_dofs, k) where k is the number of columns."""
        rng = np.random.default_rng(3)
        categories = np.asfortranarray(
            np.column_stack(
                [rng.integers(0, 10, size=200), rng.integers(0, 10, size=200)]
            ).astype(np.uint32)
        )
        k = 4
        Y = rng.standard_normal((200, k))
        solver = Solver(categories)
        batch = solver.solve_batch(Y)
        assert batch.x.shape[1] == k
        assert batch.x.shape[0] == solver.n_dofs

    def test_batch_demeaned_shape(self):
        """batch.demeaned should have shape (n_obs, k)."""
        rng = np.random.default_rng(4)
        categories = np.asfortranarray(
            np.column_stack(
                [rng.integers(0, 10, size=200), rng.integers(0, 10, size=200)]
            ).astype(np.uint32)
        )
        k = 3
        Y = rng.standard_normal((200, k))
        solver = Solver(categories)
        batch = solver.solve_batch(Y)
        assert batch.demeaned.shape == (200, k)

    def test_batch_converged_length_matches_k(self):
        """batch.converged should have length equal to number of RHS columns."""
        rng = np.random.default_rng(6)
        categories = np.asfortranarray(
            np.column_stack(
                [rng.integers(0, 10, size=200), rng.integers(0, 10, size=200)]
            ).astype(np.uint32)
        )
        k = 5
        Y = rng.standard_normal((200, k))
        solver = Solver(categories)
        batch = solver.solve_batch(Y)
        assert len(batch.converged) == k
        assert len(batch.iterations) == k
        assert len(batch.residual) == k

    def test_batch_zero_rhs_gives_zero_solution(self):
        """A batch of zero RHS vectors should give zero coefficient columns."""
        rng = np.random.default_rng(8)
        categories = np.asfortranarray(
            np.column_stack(
                [rng.integers(0, 10, size=200), rng.integers(0, 10, size=200)]
            ).astype(np.uint32)
        )
        k = 3
        Y = np.zeros((200, k))
        solver = Solver(categories)
        batch = solver.solve_batch(Y)
        for col in range(k):
            if batch.converged[col]:
                np.testing.assert_allclose(batch.x[:, col], 0.0, atol=1e-10)
