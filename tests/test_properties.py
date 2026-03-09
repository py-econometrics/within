from __future__ import annotations

import pickle

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from within import CG, OperatorRepr, Solver, solve


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
        np.testing.assert_array_equal(r1.x, r2.x)

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
