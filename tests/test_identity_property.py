"""Numerical identity property tests.

Three claims are checked on randomly generated data:

1. ``F = T_depth + T_cov + T_bio`` for every gene (Corollary 1.4).
2. The stratified estimator removes exactly ``J_A - J_B`` from the difference
   (E4's mechanism identity).
3. ``n_strata=1`` reproduces the pooled estimator.

These are properties, not examples: hypothesis generates the shapes and the
random seed, and the assertions must hold for every case.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from depthcheck.fano import fano_by_gene, stratified_delta_fano
from depthcheck.moments import K_DEFAULT, SufficientStats, implied_delta_fano


def _random_counts(seed: int, n_cells: int, n_genes: int,
                   density: float = None, mean: float = None):
    rng = np.random.default_rng(seed)
    if density is None:
        density = rng.uniform(0.15, 0.9)
    if mean is None:
        mean = rng.uniform(0.5, 8.0)
    X = (rng.random((n_cells, n_genes)) < density) * rng.poisson(
        mean, size=(n_cells, n_genes))
    X = X.astype(np.float64)
    empty = X.sum(axis=1) == 0
    if np.any(empty):
        X[empty, 0] = 1.0
    L = X.sum(axis=1)  # library size is the row sum, by definition
    return X, L


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(0, 100_000),
    n_cells=st.integers(6, 40),
    n_genes=st.integers(1, 12),
)
def test_fano_identity_holds_gene_by_gene(seed: int, n_cells: int, n_genes: int) -> None:
    X, L = _random_counts(seed, n_cells, n_genes)
    st_obj = SufficientStats.zeros(1, n_genes, K=K_DEFAULT)
    st_obj.add_cells(np.zeros(n_cells, dtype=np.int64), X, L)
    pm = st_obj.pool()
    ok = np.isfinite(pm.fano)
    assert ok.any()
    recon = pm.t_depth[ok] + pm.t_cov[ok] + pm.t_bio[ok]
    err = np.max(np.abs(pm.fano[ok] - recon)) / max(1.0, np.max(np.abs(pm.fano[ok])))
    assert err < 1e-10, f"identity error {err:.3e}"


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 100_000), n_cells=st.integers(24, 60))
def test_stratified_identity_matches_j_gap(seed: int, n_cells: int) -> None:
    X, L = _random_counts(seed, n_cells, 40, density=0.99, mean=8.0)
    is_a = np.arange(n_cells) % 2 == 0
    res = stratified_delta_fano(X, L, is_a, n_strata=3, min_cells=2)
    assert res is not None
    left = res.delta_naive - res.delta_strat
    right = res.j_gap
    ok = np.isfinite(left) & np.isfinite(right)
    assert ok.sum() > 0
    err = np.max(np.abs(left[ok] - right[ok]))
    scale = max(1.0, np.max(np.abs(res.delta_naive[ok])))
    assert err / scale < 1e-10, f"J identity error {err:.3e}"


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 100_000), n_cells=st.integers(8, 40))
def test_one_stratum_equals_pooled(seed: int, n_cells: int) -> None:
    X, L = _random_counts(seed, n_cells, 5)
    is_a = np.arange(n_cells) % 3 == 0
    res = stratified_delta_fano(X, L, is_a, n_strata=1, min_cells=1)
    assert res is not None
    fa, _ = fano_by_gene(X[is_a], L[is_a])
    fb, _ = fano_by_gene(X[~is_a], L[~is_a])
    ok = np.isfinite(fa) & np.isfinite(fb)
    assert np.allclose(res.delta_strat[ok], (fa - fb)[ok], rtol=1e-12, atol=1e-12)


def test_implied_matches_definition(seed: int = 0) -> None:
    _, L = _random_counts(seed, 30, 2)
    half = L.size // 2
    got = implied_delta_fano(L[:half], L[half:], K=37.0)
    want = 37.0 * (np.mean(1.0 / L[:half]) - np.mean(1.0 / L[half:]))
    assert abs(got - want) <= 1e-12 * max(1.0, abs(want))


def test_zero_variance_does_not_raise() -> None:
    """A perfectly balanced two-cell case must still return finite values."""
    X = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]])
    L = X.sum(axis=1)
    st_obj = SufficientStats.zeros(1, 2)
    st_obj.add_cells(np.zeros(4, dtype=np.int64), X, L)
    pm = st_obj.pool()
    assert np.all(np.isfinite(pm.t_depth))
    assert np.isfinite(pm.e1l)


def test_negative_or_zero_library_size_is_rejected() -> None:
    X = np.ones((2, 2))
    st_obj = SufficientStats.zeros(1, 2)
    with pytest.raises(ValueError):
        st_obj.add_cells(np.zeros(2, dtype=np.int64), X, np.array([1.0, 0.0]))
