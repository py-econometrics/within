"""Problem generator registry for benchmarks.

Each generator returns ``(categories, n_levels, y)`` — a list of int64
category arrays, the level counts, and a response vector ``y``.
Register new generators with ``@register_generator("key")``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

GeneratorFn = Callable[..., tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]]

_REGISTRY: dict[str, GeneratorFn] = {}


def register_generator(key: str) -> Callable[[GeneratorFn], GeneratorFn]:
    """Decorator that registers a generator function under *key*."""

    def _decorator(fn: GeneratorFn) -> GeneratorFn:
        if key in _REGISTRY:
            raise ValueError(f"Duplicate generator key: {key!r}")
        _REGISTRY[key] = fn
        return fn

    return _decorator


def get_generator(key: str) -> GeneratorFn:
    """Look up a registered generator by key."""
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown generator {key!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def list_generators() -> list[str]:
    """Return sorted list of registered generator keys."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matvec_d(
    categories: list[NDArray[np.int64]],
    n_levels: list[int],
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute D @ x where D is the categorical design matrix."""
    n_rows = len(categories[0])
    y = np.zeros(n_rows, dtype=np.float64)
    offset = 0
    for cats, nl in zip(categories, n_levels):
        y += x[offset + cats]
        offset += nl
    return y


def _make_response(
    categories: list[NDArray[np.int64]],
    n_levels: list[int],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    n_dofs = sum(n_levels)
    n_rows = len(categories[0])
    true_coeffs = rng.standard_normal(n_dofs)
    return _matvec_d(categories, n_levels, true_coeffs) + 0.1 * rng.standard_normal(n_rows)


# ===================================================================
# 3-FE generators
# ===================================================================


@register_generator("sparse_3fe")
def sparse_3fe(
    n_levels: tuple[int, int, int] = (100, 100, 100),
    edges_per_level: int = 3,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Sparse 3-FE with few edges per level — challenging coupling."""
    rng = np.random.default_rng(seed)
    cat_lists: list[list[int]] = [[], [], []]
    for factor in range(3):
        for level in range(n_levels[factor]):
            for _ in range(edges_per_level):
                cat_lists[0].append(level if factor == 0 else rng.integers(0, n_levels[0]))
                cat_lists[1].append(level if factor == 1 else rng.integers(0, n_levels[1]))
                cat_lists[2].append(level if factor == 2 else rng.integers(0, n_levels[2]))
    cats = [np.array(c, dtype=np.int64) for c in cat_lists]
    n_levels_list = list(n_levels)
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


@register_generator("chain_3fe")
def chain_3fe(
    n_levels: int = 100,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Triple chain: path-like bipartite graphs in all factor pairs."""
    rng = np.random.default_rng(seed)
    cat_a: list[int] = []
    cat_b: list[int] = []
    cat_c: list[int] = []
    for i in range(n_levels):
        cat_a.append(i); cat_b.append(i); cat_c.append(i)
        if i + 1 < n_levels:
            cat_a.append(i + 1); cat_b.append(i); cat_c.append(i)
            cat_a.append(i); cat_b.append(i + 1); cat_c.append(i)
            cat_a.append(i); cat_b.append(i); cat_c.append(i + 1)
    cats = [np.array(c, dtype=np.int64) for c in [cat_a, cat_b, cat_c]]
    n_levels_list = [n_levels] * 3
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


@register_generator("barbell_3fe")
def barbell_3fe(
    n_levels: int = 100,
    bridge_width: int = 4,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Two dense clusters connected by a narrow bridge in all three factors."""
    rng = np.random.default_rng(seed)
    half = n_levels // 2
    cluster_degree = 3
    cat_a: list[int] = []
    cat_b: list[int] = []
    cat_c: list[int] = []

    for i in range(half):
        b_nb = rng.choice(half, size=min(cluster_degree, half), replace=False)
        c_nb = rng.choice(half, size=min(cluster_degree, half), replace=False)
        for j_b, j_c in zip(b_nb, c_nb):
            cat_a.append(i); cat_b.append(int(j_b)); cat_c.append(int(j_c))

    for i in range(half, n_levels):
        b_nb = rng.choice(range(half, n_levels), size=min(cluster_degree, n_levels - half), replace=False)
        c_nb = rng.choice(range(half, n_levels), size=min(cluster_degree, n_levels - half), replace=False)
        for j_b, j_c in zip(b_nb, c_nb):
            cat_a.append(i); cat_b.append(int(j_b)); cat_c.append(int(j_c))

    for w in range(bridge_width):
        cat_a.append(half - 1 - w); cat_b.append(half + w); cat_c.append(half - 1 - w)
        cat_a.append(half - 1 - w); cat_b.append(half - 1 - w); cat_c.append(half + w)
        cat_a.append(half + w); cat_b.append(half - 1 - w); cat_c.append(half - 1 - w)

    cats = [np.array(c, dtype=np.int64) for c in [cat_a, cat_b, cat_c]]
    n_levels_list = [n_levels] * 3
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


@register_generator("imbalanced_3fe")
def imbalanced_3fe(
    n_levels: tuple[int, int, int] = (100, 100, 100),
    n_rows: int = 10000,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """3-FE with power-law level frequencies (highly imbalanced)."""
    rng = np.random.default_rng(seed)
    cats = []
    for factor in range(3):
        probs = np.arange(1, n_levels[factor] + 1, dtype=np.float64) ** (-1.5)
        probs /= probs.sum()
        cats.append(rng.choice(n_levels[factor], size=n_rows, p=probs).astype(np.int64))
    n_levels_list = list(n_levels)
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


@register_generator("clustered_3fe")
def clustered_3fe(
    n_levels: tuple[int, int, int] = (100, 100, 100),
    n_clusters: int = 5,
    obs_per_cluster: int = 100,
    bridge_obs: int = 5,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """3-FE with clustered structure (near-disconnected components)."""
    rng = np.random.default_rng(seed)
    levels_per_cluster = [n // n_clusters for n in n_levels]
    cat_lists: list[list[int]] = [[], [], []]

    for cluster in range(n_clusters):
        for _ in range(obs_per_cluster):
            for factor in range(3):
                start = cluster * levels_per_cluster[factor]
                end = start + levels_per_cluster[factor]
                cat_lists[factor].append(rng.integers(start, end))

    for cluster in range(n_clusters - 1):
        for _ in range(bridge_obs):
            for factor in range(3):
                if rng.random() < 0.5:
                    start = cluster * levels_per_cluster[factor]
                else:
                    start = (cluster + 1) * levels_per_cluster[factor]
                cat_lists[factor].append(
                    rng.integers(start, start + levels_per_cluster[factor])
                )

    cats = [np.array(c, dtype=np.int64) for c in cat_lists]
    n_levels_list = list(n_levels)
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


@register_generator("random_3fe")
def random_3fe(
    n_levels: tuple[int, int, int] = (100, 100, 100),
    n_rows: int = 10000,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Random 3-FE problem with uniform level distribution."""
    rng = np.random.default_rng(seed)
    cats = [
        rng.integers(0, n_levels[i], size=n_rows, dtype=np.int64) for i in range(3)
    ]
    n_levels_list = list(n_levels)
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


# ===================================================================
# 2-FE generators
# ===================================================================


@register_generator("chain_2fe")
def chain_2fe(
    n_levels: int = 100,
    obs_per_edge: int = 1,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Chain: A0-B0-A1-B1-A2-... — O(n^2) condition number."""
    rng = np.random.default_rng(seed)
    cat_a: list[int] = []
    cat_b: list[int] = []
    for i in range(n_levels):
        cat_a.extend([i] * obs_per_edge)
        cat_b.extend([i] * obs_per_edge)
        if i + 1 < n_levels:
            cat_a.extend([i + 1] * obs_per_edge)
            cat_b.extend([i] * obs_per_edge)
    cats = [np.array(cat_a, dtype=np.int64), np.array(cat_b, dtype=np.int64)]
    n_levels_list = [n_levels, n_levels]
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


@register_generator("star_2fe")
def star_2fe(
    n_levels: int = 100,
    obs_per_edge: int = 1,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Star: one central A level connected to all B levels."""
    rng = np.random.default_rng(seed)
    cat_a: list[int] = []
    cat_b: list[int] = []
    for j in range(n_levels):
        cat_a.extend([0] * obs_per_edge)
        cat_b.extend([j] * obs_per_edge)
    for i in range(1, n_levels):
        cat_a.extend([i] * obs_per_edge)
        cat_b.extend([rng.integers(0, n_levels)] * obs_per_edge)
    cats = [np.array(cat_a, dtype=np.int64), np.array(cat_b, dtype=np.int64)]
    n_levels_list = [n_levels, n_levels]
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


@register_generator("barbell_2fe")
def barbell_2fe(
    n_levels: int = 100,
    bridge_obs: int = 1,
    cluster_obs: int = 10,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Barbell: two dense clusters connected by thin bridge."""
    rng = np.random.default_rng(seed)
    half = n_levels // 2
    cat_a: list[int] = []
    cat_b: list[int] = []

    for i in range(half):
        for j in range(half):
            cat_a.extend([i] * cluster_obs)
            cat_b.extend([j] * cluster_obs)

    for i in range(half, n_levels):
        for j in range(half, n_levels):
            cat_a.extend([i] * cluster_obs)
            cat_b.extend([j] * cluster_obs)

    cat_a.extend([half - 1] * bridge_obs)
    cat_b.extend([half] * bridge_obs)

    cats = [np.array(cat_a, dtype=np.int64), np.array(cat_b, dtype=np.int64)]
    n_levels_list = [n_levels, n_levels]
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


@register_generator("expander_2fe")
def expander_2fe(
    n_levels: int = 100,
    degree: int = 3,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Random regular bipartite graph — good expansion."""
    rng = np.random.default_rng(seed)
    cat_a: list[int] = []
    cat_b: list[int] = []
    for i in range(n_levels):
        neighbors = rng.choice(n_levels, size=degree, replace=False)
        for j in neighbors:
            cat_a.append(i)
            cat_b.append(j)
    cats = [np.array(cat_a, dtype=np.int64), np.array(cat_b, dtype=np.int64)]
    n_levels_list = [n_levels, n_levels]
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


@register_generator("grid_2fe")
def grid_2fe(
    n_side: int = 10,
    obs_per_edge: int = 1,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Grid-like: A levels are rows, B levels are columns."""
    rng = np.random.default_rng(seed)
    cat_a: list[int] = []
    cat_b: list[int] = []
    for i in range(n_side):
        for j in range(n_side):
            cat_a.extend([i] * obs_per_edge)
            cat_b.extend([j] * obs_per_edge)
    cats = [np.array(cat_a, dtype=np.int64), np.array(cat_b, dtype=np.int64)]
    n_levels_list = [n_side, n_side]
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


# ===================================================================
# k-FE generators (generalized)
# ===================================================================


@register_generator("sparse_kfe")
def sparse_kfe(
    k: int = 3,
    n_levels: int = 30,
    edges_per_level: int = 3,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Generic k-factor sparse random design."""
    rng = np.random.default_rng(seed)
    cat_lists: list[list[int]] = [[] for _ in range(k)]
    for factor in range(k):
        for level in range(n_levels):
            for _ in range(edges_per_level):
                for q in range(k):
                    if q == factor:
                        cat_lists[q].append(level)
                    else:
                        cat_lists[q].append(int(rng.integers(0, n_levels)))
    cats = [np.array(c, dtype=np.int64) for c in cat_lists]
    n_levels_list = [n_levels] * k
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


@register_generator("chain_kfe")
def chain_kfe(
    k: int = 3,
    n_levels: int = 30,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Generic k-factor chain: path-like bipartite graphs in all pairs."""
    rng = np.random.default_rng(seed)
    cat_lists: list[list[int]] = [[] for _ in range(k)]
    for i in range(n_levels):
        for q in range(k):
            cat_lists[q].append(i)
        if i + 1 < n_levels:
            for q in range(k):
                for f in range(k):
                    cat_lists[f].append(i + 1 if f == q else i)
    cats = [np.array(c, dtype=np.int64) for c in cat_lists]
    n_levels_list = [n_levels] * k
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


# ===================================================================
# Uniform / synthetic generator (wraps library helper)
# ===================================================================


@register_generator("fixest_dgp")
def fixest_dgp(
    n_obs: int = 100_000,
    dgp_type: str = "simple",
    n_fe: int = 2,
    n_years: int = 10,
    n_indiv_per_firm: int = 23,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Fixest-style DGP: individual x year (x firm) panel data."""
    rng = np.random.default_rng(seed)

    n_indiv = max(1, round(n_obs / n_years))
    n_firm = max(1, round(n_indiv / n_indiv_per_firm))

    indiv_id = np.repeat(np.arange(n_indiv), n_years)[:n_obs].astype(np.int64)
    year = np.tile(np.arange(n_years), n_indiv)[:n_obs].astype(np.int64)

    if dgp_type == "simple":
        firm_id = rng.integers(0, n_firm, size=n_obs, dtype=np.int64)
    else:
        firm_id = np.tile(np.arange(n_firm), (n_obs // n_firm) + 1)[:n_obs].astype(
            np.int64
        )

    if n_fe == 2:
        categories = [indiv_id, year]
        n_levels_list = [n_indiv, n_years]
    else:
        categories = [indiv_id, year, firm_id]
        n_levels_list = [n_indiv, n_years, n_firm]

    # Response: y = unit_fe + year_fe + (firm_fe) + x + noise
    unit_fe = rng.standard_normal(n_indiv)
    year_fe = rng.standard_normal(n_years)
    x1 = rng.standard_normal(n_obs)
    y = x1 + unit_fe[indiv_id] + year_fe[year]
    if n_fe >= 3:
        firm_fe = rng.standard_normal(n_firm)
        y += firm_fe[firm_id]
    y += rng.standard_normal(n_obs)

    return categories, n_levels_list, y


# ===================================================================
# AKM panel generators (realistic employer-employee data)
# ===================================================================


@register_generator("akm_power_law")
def akm_power_law(
    n_workers: int = 10_000,
    n_firms: int = 500,
    n_years: int = 10,
    zipf_exponent: float = 1.5,
    mobility_rate: float = 0.15,
    n_fe: int = 3,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """AKM panel with Zipf firm-size distribution (extreme degree heterogeneity)."""
    from ._akm_helpers import panel_to_design, simulate_mobility, zipf_firm_sizes

    rng = np.random.default_rng(seed)
    firm_weights = zipf_firm_sizes(n_firms, zipf_exponent)
    initial_firm = rng.choice(n_firms, size=n_workers, p=firm_weights).astype(np.intp)
    assignments = simulate_mobility(
        initial_firm, n_years, mobility_rate, firm_weights, rng,
    )
    return panel_to_design(assignments, n_fe, rng)


@register_generator("akm_low_mobility")
def akm_low_mobility(
    n_workers: int = 10_000,
    n_firms: int = 500,
    n_years: int = 10,
    annual_mobility_rate: float = 0.05,
    n_fe: int = 3,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """AKM panel with very low mobility (sparse bipartite worker-firm graph)."""
    from ._akm_helpers import panel_to_design, simulate_mobility

    rng = np.random.default_rng(seed)
    # Uniform firm sizes to isolate mobility effect
    firm_weights = np.ones(n_firms, dtype=np.float64) / n_firms
    initial_firm = rng.integers(0, n_firms, size=n_workers, dtype=np.intp)
    assignments = simulate_mobility(
        initial_firm, n_years, annual_mobility_rate, firm_weights, rng,
    )
    return panel_to_design(assignments, n_fe, rng)


@register_generator("akm_disconnected")
def akm_disconnected(
    n_workers: int = 10_000,
    n_firms: int = 500,
    n_years: int = 10,
    n_clusters: int = 5,
    within_mobility: float = 0.10,
    cross_cluster_rate: float = 0.01,
    prune_components: bool = True,
    n_fe: int = 3,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """AKM panel with near-disconnected regional clusters."""
    from ._akm_helpers import panel_to_design, simulate_mobility

    rng = np.random.default_rng(seed)
    # Uniform firm sizes, partitioned into clusters
    firm_weights = np.ones(n_firms, dtype=np.float64) / n_firms
    cluster_map = np.repeat(
        np.arange(n_clusters, dtype=np.intp),
        (n_firms + n_clusters - 1) // n_clusters,
    )[:n_firms]
    # Workers start in a random firm within a random cluster
    initial_firm = rng.integers(0, n_firms, size=n_workers, dtype=np.intp)
    assignments = simulate_mobility(
        initial_firm, n_years, within_mobility, firm_weights, rng,
        cluster_map=cluster_map, cross_cluster_rate=cross_cluster_rate,
    )
    return panel_to_design(
        assignments, n_fe, rng, prune_components=prune_components,
    )


@register_generator("akm_realistic")
def akm_realistic(
    n_workers: int = 10_000,
    n_firms: int = 500,
    n_years: int = 10,
    zipf_exponent: float = 1.3,
    annual_mobility_rate: float = 0.10,
    n_clusters: int = 5,
    cross_cluster_rate: float = 0.02,
    prune_components: bool = True,
    n_fe: int = 3,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """AKM panel combining all pathologies: power-law sizes + low mobility + clusters."""
    from ._akm_helpers import panel_to_design, simulate_mobility, zipf_firm_sizes

    rng = np.random.default_rng(seed)
    firm_weights = zipf_firm_sizes(n_firms, zipf_exponent)
    cluster_map = np.repeat(
        np.arange(n_clusters, dtype=np.intp),
        (n_firms + n_clusters - 1) // n_clusters,
    )[:n_firms]
    initial_firm = rng.choice(n_firms, size=n_workers, p=firm_weights).astype(np.intp)
    assignments = simulate_mobility(
        initial_firm, n_years, annual_mobility_rate, firm_weights, rng,
        cluster_map=cluster_map, cross_cluster_rate=cross_cluster_rate,
    )
    return panel_to_design(
        assignments, n_fe, rng, prune_components=prune_components,
    )


@register_generator("random_kfe")
def random_kfe(
    k: int = 4,
    n_levels_per_factor: list[int] | None = None,
    n_rows: int = 10000,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Random k-FE with uniform level distribution (generalized random_3fe)."""
    rng = np.random.default_rng(seed)
    if n_levels_per_factor is None:
        n_levels_per_factor = [50] * k
    assert len(n_levels_per_factor) == k
    cats = [
        rng.integers(0, nl, size=n_rows, dtype=np.int64)
        for nl in n_levels_per_factor
    ]
    return cats, list(n_levels_per_factor), _make_response(cats, n_levels_per_factor, rng)


@register_generator("imbalanced_kfe")
def imbalanced_kfe(
    k: int = 4,
    n_levels_per_factor: list[int] | None = None,
    n_rows: int = 10000,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """k-FE with power-law level frequencies (highly imbalanced)."""
    rng = np.random.default_rng(seed)
    if n_levels_per_factor is None:
        n_levels_per_factor = [50] * k
    assert len(n_levels_per_factor) == k
    cats = []
    for nl in n_levels_per_factor:
        probs = np.arange(1, nl + 1, dtype=np.float64) ** (-1.5)
        probs /= probs.sum()
        cats.append(rng.choice(nl, size=n_rows, p=probs).astype(np.int64))
    return cats, list(n_levels_per_factor), _make_response(cats, n_levels_per_factor, rng)


@register_generator("disconnected_kfe")
def disconnected_kfe(
    k: int = 3,
    n_levels: int = 100,
    n_clusters: int = 10,
    obs_per_cluster: int = 200,
    bridge_obs: int = 0,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """k-FE with many (near-)disconnected components.

    Each cluster uses a disjoint slice of levels per factor.
    *bridge_obs* controls how many observations connect adjacent clusters
    (0 = truly disconnected components).
    """
    rng = np.random.default_rng(seed)
    levels_per_cluster = n_levels // n_clusters
    cat_lists: list[list[int]] = [[] for _ in range(k)]

    for cluster in range(n_clusters):
        start = cluster * levels_per_cluster
        end = start + levels_per_cluster
        for _ in range(obs_per_cluster):
            for q in range(k):
                cat_lists[q].append(int(rng.integers(start, end)))

    # Bridge observations between adjacent clusters
    for cluster in range(n_clusters - 1):
        for _ in range(bridge_obs):
            for q in range(k):
                # Pick from either cluster
                if rng.random() < 0.5:
                    start = cluster * levels_per_cluster
                else:
                    start = (cluster + 1) * levels_per_cluster
                cat_lists[q].append(int(rng.integers(start, start + levels_per_cluster)))

    cats = [np.array(c, dtype=np.int64) for c in cat_lists]
    n_levels_list = [n_levels] * k
    return cats, n_levels_list, _make_response(cats, n_levels_list, rng)


@register_generator("uniform_kfe")
def uniform_kfe(
    n_levels_per_factor: list[int] | None = None,
    n_rows: int = 10000,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Uniform random categories via ``generate_synthetic_data``."""
    from within import generate_synthetic_data

    if n_levels_per_factor is None:
        n_levels_per_factor = [50, 80, 30]
    cats, _true_coeffs, y = generate_synthetic_data(
        n_levels_per_factor, n_rows, seed=seed,
    )
    return cats, n_levels_per_factor, y
