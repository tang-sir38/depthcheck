"""Synthetic data with a known ``Delta F``.

Two ground truths are constructed from the same composition vector:

``depth only``
    Groups differ only in library size. The lemma says
    ``Delta F = K Delta E[1/L]`` with no residual, so the verdict must be FAIL:
    the contrast is fully explained by the depth term.
``noise only``
    Groups share the exact same library sizes and differ only in
    overdispersion. The implied term is zero and the residual is large, so the
    verdict must be PASS.

If either test fails, the estimator or the verdict rule is wrong -- these are
the two cases where the right answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from depthcheck.cli import _bootstrap
from depthcheck.decompose import FAIL, PASS, decompose
from depthcheck.fano import fano_by_gene
from depthcheck.moments import K_DEFAULT, SufficientStats, implied_delta_fano

from _synth import two_group_dataset


def _stats_by_donor(X: np.ndarray, L: np.ndarray, donor: np.ndarray) -> SufficientStats:
    levels = sorted(set(donor.tolist()))
    stats = SufficientStats.zeros(len(levels), X.shape[1], K=K_DEFAULT)
    for i, level in enumerate(levels):
        rows = donor == level
        stats.add_cells(np.full(int(rows.sum()), i, dtype=np.int64), X[rows], L[rows])
    return stats


def _analyse(X, L, is_a, donor):
    levels = sorted(set(donor.tolist()))
    lookup = {d: i for i, d in enumerate(levels)}
    block = np.array([lookup[d] for d in donor], dtype=np.int64)
    stats = _stats_by_donor(X, L, donor)
    idx_a = np.unique(block[is_a])
    idx_b = np.unique(block[~is_a])

    fa, pba = fano_by_gene(X[is_a], L[is_a])
    fb, pbb = fano_by_gene(X[~is_a], L[~is_a])
    mask = (K_DEFAULT * pba >= 0.5) & (K_DEFAULT * pbb >= 0.5) \
        & np.isfinite(fa) & np.isfinite(fb)
    observed = float(np.median((fa - fb)[mask]))
    implied = implied_delta_fano(L[is_a], L[~is_a])
    boot = _bootstrap(stats, idx_a, idx_b, mask, n_boot=200, seed=13)
    dec = decompose(
        observed=observed,
        implied=implied,
        implied_ci=boot["implied"],
        residual_ci=boot["residual"],
        observed_ci=boot["observed"],
    )
    return observed, implied, mask.sum(), dec


def test_depth_only_contrast_is_fully_explained() -> None:
    X, L, is_a, donor = two_group_dataset(
        n_cells=3000, n_genes=300, depth_scale_a=0.75, seed=5
    )
    observed, implied, n_genes, dec = _analyse(X, L, is_a, donor)
    assert n_genes > 150
    assert abs(observed - implied) < 0.06, (observed, implied)
    assert dec.residual_ci[0] <= 0.0 <= dec.residual_ci[1]
    assert dec.verdict == FAIL


def test_noise_only_contrast_is_recovered() -> None:
    X, L, is_a, donor = two_group_dataset(
        n_cells=3000, n_genes=300, depth_scale_a=1.0, alpha_a=0.45, alpha_b=0.05,
        seed=8,
    )
    observed, implied, n_genes, dec = _analyse(X, L, is_a, donor)
    assert n_genes > 150
    assert abs(implied) < 0.02, implied  # identical library sizes by construction
    assert observed > 1.0
    assert dec.residual_ci[0] > 0.05
    assert dec.verdict == PASS


def test_implied_sign_follows_depth_direction() -> None:
    X, L, is_a, donor = two_group_dataset(
        n_cells=1200, n_genes=120, depth_scale_a=1.4, seed=11
    )
    _, implied, _, _ = _analyse(X, L, is_a, donor)
    assert implied < 0  # A is deeper, so E[1/L] is smaller in A


def test_negative_depth_scale_is_rejected() -> None:
    with pytest.raises(ValueError):
        implied_delta_fano([1.0, 2.0], [-1.0, 2.0])


def test_permutation_null_is_calibrated() -> None:
    """A real depth contrast must look extreme to the label permutation.

    Regression guard for a bug found during release: the null divided the
    sufficient statistics by the number of *blocks* instead of the number of
    *cells*, which inflated every Fano value by the mean block size and made a
    clear depth effect look ordinary (p = 0.74 instead of p = 0.008).
    """
    from depthcheck.fano import permutation_null_delta

    X, L, is_a, donor = two_group_dataset(
        n_cells=1200, n_genes=120, depth_scale_a=0.7, seed=5
    )
    levels = sorted(set(donor.tolist()))
    lookup = {d: i for i, d in enumerate(levels)}
    block = np.array([lookup[d] for d in donor], dtype=np.int64)
    stats = _stats_by_donor(X, L, donor)
    idx_a = np.unique(block[is_a])
    idx_b = np.unique(block[~is_a])
    sub = stats.subset(np.concatenate([idx_a, idx_b]))
    is_a_blocks = np.array([True] * idx_a.size + [False] * idx_b.size)

    fa, pba = fano_by_gene(X[is_a], L[is_a])
    fb, pbb = fano_by_gene(X[~is_a], L[~is_a])
    mask = (K_DEFAULT * pba >= 0.5) & (K_DEFAULT * pbb >= 0.5) \
        & np.isfinite(fa) & np.isfinite(fb)

    res = permutation_null_delta(sub, is_a_blocks, gene_mask=mask,
                                 exact_max=100000, seed=0)
    assert res.method == "exact"
    assert abs(res.observed - float(np.median((fa - fb)[mask]))) < 1e-9
    assert res.p_value < 0.05


def test_blockwise_stratified_matches_cellwise() -> None:
    """The CLI accumulates per (group, stratum) blocks; the API works per cell.

    Both must return the same depth-balanced difference, otherwise the two
    code paths have drifted.
    """
    from depthcheck.fano import rank_bins, stratified_delta_fano

    X, L, is_a, _ = two_group_dataset(
        n_cells=800, n_genes=40, depth_scale_a=0.8, seed=3
    )
    res = stratified_delta_fano(X, L, is_a, n_strata=4, min_cells=10)
    assert res is not None

    bins = rank_bins(L, 4)
    num = np.zeros(X.shape[1])
    w_sum = 0.0
    for k in range(4):
        sel_a = (bins == k) & is_a
        sel_b = (bins == k) & (~is_a)
        if sel_a.sum() < 10 or sel_b.sum() < 10:
            continue
        fa, _ = fano_by_gene(X[sel_a], L[sel_a])
        fb, _ = fano_by_gene(X[sel_b], L[sel_b])
        w = float(sel_a.sum() + sel_b.sum())
        num += w * (fa - fb)
        w_sum += w
    block_delta = num / w_sum
    ok = np.isfinite(block_delta) & np.isfinite(res.delta_strat)
    assert ok.sum() > 0
    assert np.allclose(block_delta[ok], res.delta_strat[ok], rtol=1e-10, atol=1e-10)
