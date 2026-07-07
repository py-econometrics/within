"""Single varying slope on one factor (#59), cross-checked against pyfixest.

pyfixest has no native ``f[z]`` syntax; the reference is the explicit
interaction parametrization ``y ~ C(f) + i(f, z) - 1``, which estimates the
same per-level intercepts and slopes with no absorption or reference coding.
"""

import numpy as np
import pytest

import within
from within import Effect, LsmrOptions

pf = pytest.importorskip("pyfixest")
pd = pytest.importorskip("pandas")

OPTS = LsmrOptions(tol=1e-12, maxiter=2000)
N_LEVELS = 6


def _panel(seed, n=400):
    rng = np.random.default_rng(seed)
    f = rng.integers(0, N_LEVELS, n).astype(np.uint32)
    z = rng.normal(size=n)
    a = rng.normal(size=N_LEVELS)
    b = rng.normal(size=N_LEVELS)
    y = a[f] + b[f] * z + rng.normal(scale=0.1, size=n)
    return f, z, y


def _pyfixest_coef(f, z, y, formula, w=None):
    df = pd.DataFrame({"f": [f"L{v}" for v in f], "z": z, "y": y})
    weights = None
    if w is not None:
        df["w"] = w
        weights = "w"
    return pf.feols(formula, data=df, weights=weights).coef()


def _assert_matches(res, coef, *, intercepts=True, skip_slopes=()):
    base = N_LEVELS if intercepts else 0
    for lvl in range(N_LEVELS):
        if intercepts:
            assert res.x[lvl] == pytest.approx(
                coef[f"C(f)[L{lvl}]"], rel=1e-6, abs=1e-8
            )
        if lvl not in skip_slopes:
            assert res.x[base + lvl] == pytest.approx(
                coef[f"f::L{lvl}:z"], rel=1e-6, abs=1e-8
            )


def test_single_slope_matches_pyfixest():
    f, z, y = _panel(seed=42)
    res = within.solve([Effect(f, True, [z])], y, OPTS)
    assert res.converged
    assert res.unidentified == []

    _assert_matches(res, _pyfixest_coef(f, z, y, "y ~ C(f) + i(f, z) - 1"))


def test_weighted_single_slope_matches_pyfixest():
    f, z, y = _panel(seed=7)
    w = np.random.default_rng(11).uniform(0.2, 3.0, size=len(y))
    res = within.solve([Effect(f, True, [z])], y, OPTS, weights=w)
    assert res.converged

    _assert_matches(res, _pyfixest_coef(f, z, y, "y ~ C(f) + i(f, z) - 1", w=w))


def test_slope_only_term_matches_pyfixest():
    f, z, y = _panel(seed=3)
    res = within.solve([Effect(f, False, [z])], y, OPTS)
    assert res.converged
    assert res.unidentified == []
    assert len(res.x) == N_LEVELS

    _assert_matches(res, _pyfixest_coef(f, z, y, "y ~ i(f, z) - 1"), intercepts=False)


def test_constant_slope_level_reports_drop_with_exact_zero():
    f, z, y = _panel(seed=5)
    z = z.copy()
    z[f == 1] = 2.5
    res = within.solve([Effect(f, True, [z])], y, OPTS)

    assert res.unidentified == [(0, 1, 1)]
    assert res.x[N_LEVELS + 1] == 0.0
    assert np.isfinite(res.x).all()

    # The un-dropped coefficients still match the reference, which drops the
    # collinear column on its own.
    _assert_matches(
        res, _pyfixest_coef(f, z, y, "y ~ C(f) + i(f, z) - 1"), skip_slopes={1}
    )
