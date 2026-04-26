from __future__ import annotations

import pickle

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from within import CG, GMRES, OperatorRepr, Preconditioner, Solver, solve
from within import AdditiveSchwarz, MultiplicativeSchwarz, ReductionStrategy


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
    def test_residual_orthogonality(self, data):
        """D^T * (y - D*x) should be approximately zero."""
        categories, y = data
        result = solve(categories, y)
        if not result.converged:
            return  # skip non-converged cases

        n_obs, n_factors = categories.shape
        residual = (
            y - result.demeaned - y
        )  # demeaned = y - D*x, so residual = D*x... wait
        # Actually: demeaned = y - D*x, so D*x = y - demeaned
        # residual = y - D*x = demeaned
        residual = result.demeaned  # this IS y - D*x

        # Check D^T * residual ≈ 0
        offset = 0
        for f in range(n_factors):
            col = categories[:, f]
            n_levels = int(col.max()) + 1
            for lvl in range(n_levels):
                mask = col == lvl
                dot = residual[mask].sum()
                assert abs(dot) < 1e-3, f"factor {f}, level {lvl}: D^T r = {dot}"
            offset += n_levels

    @given(data=random_fe_problem())
    @settings(
        max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_preconditioner_pickle_preserves_apply(self, data):
        """Pickle roundtrip of preconditioner preserves apply()."""
        categories, y = data
        solver = Solver(categories)
        precond = solver.preconditioner()
        if precond is None:
            return

        x = np.random.randn(precond.nrows)
        result_before = precond.apply(x)

        data_bytes = pickle.dumps(precond)
        precond2 = pickle.loads(data_bytes)
        result_after = precond2.apply(x)

        np.testing.assert_array_equal(result_before, result_after)

    @given(data=random_fe_problem())
    @settings(
        max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_explicit_vs_implicit_equivalent(self, data):
        """Both OperatorRepr modes should give the same solution."""
        categories, y = data
        r_implicit = solve(categories, y, CG(operator=OperatorRepr.Implicit))
        r_explicit = solve(categories, y, CG(operator=OperatorRepr.Explicit))

        if not (r_implicit.converged and r_explicit.converged):
            return

        np.testing.assert_allclose(r_implicit.x, r_explicit.x, atol=1e-6)

    @given(data=random_fe_problem())
    @settings(
        max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_unit_weights_match_unweighted(self, data):
        """weights=ones should match no weights."""
        categories, y = data
        n_obs = len(y)
        r_no_weights = solve(categories, y)
        r_unit_weights = solve(categories, y, weights=np.ones(n_obs))

        if not (r_no_weights.converged and r_unit_weights.converged):
            return

        np.testing.assert_allclose(r_no_weights.x, r_unit_weights.x, atol=1e-6)

    @given(data=random_fe_problem())
    @settings(
        max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_solve_vs_solver_equivalence(self, data):
        """solve() and Solver().solve() produce identical results."""
        categories, y = data
        r1 = solve(categories, y)
        solver = Solver(categories)
        r2 = solver.solve(y)
        if r1.converged and r2.converged:
            np.testing.assert_allclose(r1.x, r2.x, atol=1e-10)

    @given(data=random_fe_problem())
    @settings(
        max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_demeaned_is_y_minus_design_times_x(self, data):
        """demeaned[i] should equal y[i] minus the sum of factor effects for obs i."""
        categories, y = data
        result = solve(categories, y)
        if not result.converged:
            return

        n_obs, n_factors = categories.shape
        # Reconstruct D*x from categories and result.x
        fitted = np.zeros(n_obs)
        offset = 0
        for f in range(n_factors):
            col = categories[:, f]
            n_levels = int(col.max()) + 1
            fitted += result.x[offset + col]
            offset += n_levels

        expected_demeaned = y - fitted
        np.testing.assert_allclose(result.demeaned, expected_demeaned, atol=1e-8)

    @given(data=random_fe_problem())
    @settings(
        max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_solver_n_dofs_matches_x_length(self, data):
        """Solver.n_dofs should equal len(result.x)."""
        categories, y = data
        solver = Solver(categories)
        result = solver.solve(y)
        assert solver.n_dofs == len(result.x)

    @given(data=random_fe_problem())
    @settings(
        max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_solver_n_obs_matches_y_length(self, data):
        """Solver.n_obs should equal len(y)."""
        categories, y = data
        solver = Solver(categories)
        assert solver.n_obs == len(y)

    @given(data=random_fe_problem())
    @settings(
        max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_residual_nonnegative(self, data):
        """The final residual norm must be non-negative."""
        categories, y = data
        result = solve(categories, y)
        assert result.residual >= 0.0

    @given(data=random_fe_problem())
    @settings(
        max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_iterations_positive_when_not_trivial(self, data):
        """Iterations should be >= 1 for non-trivial problems."""
        categories, y = data
        if np.allclose(y, 0.0):
            return
        result = solve(categories, y)
        assert result.iterations >= 1

    @given(data=random_fe_problem())
    @settings(
        max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_time_fields_nonnegative(self, data):
        """All timing fields should be non-negative."""
        categories, y = data
        result = solve(categories, y)
        assert result.time_total >= 0.0
        assert result.time_setup >= 0.0
        assert result.time_solve >= 0.0


class TestAdvancedPreconditioners:
    """Tests for AdditiveSchwarz and MultiplicativeSchwarz preconditioner configs."""

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

    def test_multiplicative_schwarz_object_converges_with_gmres(self):
        """MultiplicativeSchwarz() with GMRES should converge."""
        rng = np.random.default_rng(42)
        categories = np.asfortranarray(
            np.column_stack(
                [rng.integers(0, 20, size=500), rng.integers(0, 20, size=500)]
            ).astype(np.uint32)
        )
        y = rng.standard_normal(500)
        result = solve(categories, y, GMRES(), preconditioner=MultiplicativeSchwarz())
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

    def test_additive_schwarz_diagnostics_available(self):
        """FePreconditioner built with additive Schwarz should expose diagnostics."""
        rng = np.random.default_rng(5)
        categories = np.asfortranarray(
            np.column_stack(
                [rng.integers(0, 10, size=300), rng.integers(0, 10, size=300)]
            ).astype(np.uint32)
        )
        solver = Solver(categories, preconditioner=Preconditioner.Additive)
        precond = solver.preconditioner()
        assert precond is not None
        diag = precond.additive_schwarz_diagnostics()
        assert diag is not None
        assert diag.total_inner_parallel_work >= 0
        assert diag.total_scatter_dofs >= 0


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
