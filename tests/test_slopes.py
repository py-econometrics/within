"""Varying slopes (#59, #60) and cross-factor routing (#61, #62), cross-checked
against pyfixest.

pyfixest has no native ``f[z]`` syntax; the reference is the explicit
interaction parametrization ``y ~ C(f) + i(f, z) - 1``, which estimates the
same per-level intercepts and slopes with no absorption or reference coding.
With a second factor the intercept blocks pick up gauge freedom, so
two-factor comparisons check residuals and slope coefficients only.
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
    res = within.solve([Effect(f, True, [z])], y, options=OPTS)
    assert res.converged
    assert res.unidentified == []

    _assert_matches(res, _pyfixest_coef(f, z, y, "y ~ C(f) + i(f, z) - 1"))


def test_weighted_single_slope_matches_pyfixest():
    f, z, y = _panel(seed=7)
    w = np.random.default_rng(11).uniform(0.2, 3.0, size=len(y))
    res = within.solve([Effect(f, True, [z])], y, options=OPTS, weights=w)
    assert res.converged

    _assert_matches(res, _pyfixest_coef(f, z, y, "y ~ C(f) + i(f, z) - 1", w=w))


def test_slope_only_term_matches_pyfixest():
    f, z, y = _panel(seed=3)
    res = within.solve([Effect(f, False, [z])], y, options=OPTS)
    assert res.converged
    assert res.unidentified == []
    assert len(res.x) == N_LEVELS

    _assert_matches(res, _pyfixest_coef(f, z, y, "y ~ i(f, z) - 1"), intercepts=False)


def test_constant_slope_level_reports_drop_with_exact_zero():
    f, z, y = _panel(seed=5)
    z = z.copy()
    z[f == 1] = 2.5
    res = within.solve([Effect(f, True, [z])], y, options=OPTS)

    (u,) = res.unidentified
    assert (u.term, u.level, u.column) == (0, 1, 1)
    assert repr(u) == "UnidentifiedDirection(term=0, level=1, column=1)"
    # Parity with the (term, level, column) tuple it replaced: value-comparable
    # and hashable (the property rebuilds the record, so this is cross-instance).
    other = res.unidentified[0]
    assert u == other and hash(u) == hash(other)
    assert res.x[N_LEVELS + 1] == 0.0
    assert np.isfinite(res.x).all()

    # The un-dropped coefficients still match the reference, which drops the
    # collinear column on its own.
    _assert_matches(
        res, _pyfixest_coef(f, z, y, "y ~ C(f) + i(f, z) - 1"), skip_slopes={1}
    )


def _panel_multi(seed, v, n=400):
    rng = np.random.default_rng(seed)
    f = rng.integers(0, N_LEVELS, n).astype(np.uint32)
    base = rng.normal(size=n)
    # Correlated slopes so the raw per-level Grams are ill-conditioned.
    zs = [base * (0.6 + 0.2 * j) + 0.15 * rng.normal(size=n) for j in range(v)]
    a = rng.normal(size=N_LEVELS)
    b = rng.normal(size=(v, N_LEVELS))
    y = a[f] + sum(b[j][f] * zs[j] for j in range(v)) + rng.normal(scale=0.1, size=n)
    return f, zs, y


def _pyfixest_coef_multi(f, zs, y, w=None):
    df = pd.DataFrame({"f": [f"L{v}" for v in f], "y": y})
    for j, z in enumerate(zs):
        df[f"z{j}"] = z
    weights = None
    if w is not None:
        df["w"] = w
        weights = "w"
    terms = " + ".join(f"i(f, z{j})" for j in range(len(zs)))
    return pf.feols(f"y ~ C(f) + {terms} - 1", data=df, weights=weights).coef()


def test_three_correlated_weighted_slopes_match_pyfixest():
    f, zs, y = _panel_multi(seed=19, v=3)
    w = np.random.default_rng(29).uniform(0.2, 3.0, size=len(y))
    res = within.solve([Effect(f, True, zs)], y, options=OPTS, weights=w)
    assert res.converged
    assert res.unidentified == []

    coef = _pyfixest_coef_multi(f, zs, y, w=w)
    for lvl in range(N_LEVELS):
        assert res.x[lvl] == pytest.approx(coef[f"C(f)[L{lvl}]"], rel=1e-6, abs=1e-8)
        for j in range(3):
            assert res.x[N_LEVELS * (1 + j) + lvl] == pytest.approx(
                coef[f"f::L{lvl}:z{j}"], rel=1e-6, abs=1e-8
            )


def test_two_factor_slope_matches_pyfixest():
    rng = np.random.default_rng(101)
    n = 500
    f = rng.integers(0, N_LEVELS, n).astype(np.uint32)
    g = rng.integers(0, 2, n).astype(np.uint32)
    z = rng.normal(size=n)
    a = rng.normal(size=N_LEVELS)
    b = rng.normal(size=N_LEVELS)
    c = rng.normal(size=2)
    y = a[f] + b[f] * z + c[g] + rng.normal(scale=0.1, size=n)

    res = within.solve([Effect(f, True, [z]), Effect(g, True)], y, options=OPTS)
    assert res.converged
    assert res.unidentified == []

    df = pd.DataFrame(
        {"f": [f"L{v}" for v in f], "g": [f"G{v}" for v in g], "z": z, "y": y}
    )
    fit = pf.feols("y ~ C(f) + i(f, z) + C(g) - 1", data=df)
    coef = fit.coef()
    # A generic z keeps every slope identified; only intercepts carry gauge.
    for lvl in range(N_LEVELS):
        assert res.x[N_LEVELS + lvl] == pytest.approx(
            coef[f"f::L{lvl}:z"], rel=1e-6, abs=1e-8
        )
    assert np.allclose(res.demeaned, fit.resid(), rtol=1e-6, atol=1e-8)


def test_unit_trends_time_effects_boundary_matches_pyfixest():
    rng = np.random.default_rng(202)
    n_units, n_times = 20, 9
    unit = np.repeat(np.arange(n_units, dtype=np.uint32), n_times)
    time = np.tile(np.arange(n_times, dtype=np.uint32), n_units)
    t = time.astype(np.float64)
    alpha = rng.normal(size=n_units)
    trend = rng.normal(scale=0.3, size=n_units)
    delta = rng.normal(size=n_times)
    y = (
        alpha[unit]
        + trend[unit] * t
        + delta[time]
        + rng.normal(scale=0.1, size=n_units * n_times)
    )

    res = within.solve([Effect(unit, True, [t]), Effect(time, True)], y, options=OPTS)
    assert res.converged
    assert res.unidentified == []

    df = pd.DataFrame(
        {"f": [f"U{u}" for u in unit], "g": [f"T{k}" for k in time], "z": t, "y": y}
    )
    fit = pf.feols("y ~ C(f) + i(f, z) + C(g) - 1", data=df)
    assert np.allclose(res.demeaned, fit.resid(), rtol=1e-6, atol=1e-8)

    # The trend column is a linear combination of the time dummies, so the
    # gauge shifts every unit slope uniformly: compare slope differences
    # against whichever slope columns the reference kept.
    coef = fit.coef()
    kept = {u: coef[f"f::U{u}:z"] for u in range(n_units) if f"f::U{u}:z" in coef.index}
    assert len(kept) >= n_units - 1
    (u0, s0), *rest = sorted(kept.items())
    for u, s in rest:
        assert res.x[n_units + u] - res.x[n_units + u0] == pytest.approx(
            s - s0, rel=1e-6, abs=1e-8
        )


def test_frustrated_two_factor_slope_matches_pyfixest():
    # A partner factor with three or more levels generically frustrates the
    # (f-slope, g) pair — whitened slope rows are zero-sum, so cycles close
    # with negative sign. Solved through the Gremban double cover; the fit
    # must still match the reference exactly.
    rng = np.random.default_rng(303)
    n = 500
    f = rng.integers(0, N_LEVELS, n).astype(np.uint32)
    g = rng.integers(0, 5, n).astype(np.uint32)
    z = rng.normal(size=n)
    a = rng.normal(size=N_LEVELS)
    b = rng.normal(size=N_LEVELS)
    c = rng.normal(size=5)
    y = a[f] + b[f] * z + c[g] + rng.normal(scale=0.1, size=n)

    res = within.solve([Effect(f, True, [z]), Effect(g, True)], y, options=OPTS)
    assert res.converged
    assert res.unidentified == []

    df = pd.DataFrame(
        {"f": [f"L{v}" for v in f], "g": [f"G{v}" for v in g], "z": z, "y": y}
    )
    fit = pf.feols("y ~ C(f) + i(f, z) + C(g) - 1", data=df)
    coef = fit.coef()
    for lvl in range(N_LEVELS):
        assert res.x[N_LEVELS + lvl] == pytest.approx(
            coef[f"f::L{lvl}:z"], rel=1e-6, abs=1e-8
        )
    assert np.allclose(res.demeaned, fit.resid(), rtol=1e-6, atol=1e-8)
