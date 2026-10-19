"""Fano estimators and the label-permutation null.

Three estimators live here:

``fano_by_gene``
    The ordinary (pooled) Fano factor per gene.
``stratified_delta_fano``
    The depth-balanced estimator: cells are ranked on pooled library size,
    split into ``n_strata`` equal-count strata, Fano is computed inside each
    stratum, and strata are recombined with pooled cell-count weights. Inside a
    stratum ``E[1/L]`` is nearly constant, so the depth term cancels in the
    difference. The recombination is *not* free: it removes between-stratum
    composition, and the gap between the pooled and stratified estimates is
    exactly ``J_A - J_B`` (verified to machine precision in E4).
``permutation_null_delta``
    The null distribution of ``median_g Delta F`` under exchangeable block
    labels. Enumeration is exact when the number of distinct assignments is at
    most ``exact_max``; otherwise Monte Carlo with a fixed seed. The method is
    always reported alongside the p-value.

A note on what the permutation test can and cannot say: it tests whether the
observed contrast is compatible with exchangeable blocks (donors). It does not
test whether the contrast is biological, because depth is part of the block
label and is therefore also permuted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix, issparse

from .moments import K_DEFAULT, SufficientStats


def fano_by_gene(X, L: Sequence[float], K: float = K_DEFAULT):
    """Pooled Fano factor and mean abundance per gene.

    Parameters
    ----------
    X : scipy.sparse matrix or ndarray, shape (n_cells, n_genes)
        Raw counts.
    L : sequence of float, shape (n_cells,)
        Per-cell library size.
    K : float

    Returns
    -------
    fano : ndarray, shape (n_genes,)
    pbar : ndarray, shape (n_genes,)
    """
    Xc = X.tocsr() if issparse(X) else csr_matrix(np.asarray(X))
    st = SufficientStats.zeros(1, Xc.shape[1], K=K)
    st.add_cells(np.zeros(Xc.shape[0], dtype=np.int64), Xc, np.asarray(L, dtype=float))
    pm = st.pool()
    return pm.fano, pm.pbar


def rank_bins(L: Sequence[float], n_strata: int) -> np.ndarray:
    """Equal-count strata from a ranking on ``L`` (stable, deterministic)."""
    arr = np.asarray(L, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("L must be a non-empty 1-D array")
    if n_strata < 1:
        raise ValueError("n_strata must be >= 1")
    if n_strata > arr.size:
        raise ValueError("n_strata cannot exceed the number of cells")
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(arr.size, dtype=np.int64)
    ranks[order] = np.arange(arr.size, dtype=np.int64)
    return (ranks * n_strata) // arr.size


@dataclass
class StratifiedResult:
    """Outcome of the depth-balanced estimator for one unit.

    Attributes
    ----------
    delta_naive : ndarray
        ``F_A - F_B`` on all cells.
    delta_strat : ndarray
        Depth-balanced ``F_A - F_B``.
    f_a_strat, f_b_strat : ndarray
        Stratum-weighted Fano per group.
    j_a, j_b : ndarray
        Between-stratum composition penalty per group,
        ``F_pooled - F_strat``.
    coverage : float
        Share of cells that fell in retained strata.
    strata_used : int
        Number of retained strata.
    n_cells_used : int
    """

    delta_naive: np.ndarray
    delta_strat: np.ndarray
    f_a_strat: np.ndarray
    f_b_strat: np.ndarray
    j_a: np.ndarray
    j_b: np.ndarray
    coverage: float
    strata_used: int
    n_cells_used: int

    @property
    def j_gap(self) -> np.ndarray:
        """``J_A - J_B``; equals ``delta_naive - delta_strat`` by construction."""
        return self.j_a - self.j_b


def stratified_delta_fano(
    X,
    L: Sequence[float],
    is_a: Sequence[bool],
    n_strata: int = 5,
    min_cells: int = 15,
    K: float = K_DEFAULT,
) -> Optional[StratifiedResult]:
    """Depth-balanced difference in Fano factors.

    Cells are ranked on the *pooled* library size, so both groups share the same
    stratum boundaries. A stratum is retained only when each group contributes
    at least ``min_cells`` cells; otherwise the contrast has no common support
    there and including it would compare different parts of the depth
    distribution.

    Returns ``None`` when no stratum is retained (explicitly not estimable, not
    silently zero).
    """
    Xc = X.tocsr() if issparse(X) else csr_matrix(np.asarray(X))
    L = np.asarray(L, dtype=np.float64)
    is_a = np.asarray(is_a, dtype=bool)
    if Xc.shape[0] != L.size or is_a.size != L.size:
        raise ValueError("X, L and is_a must have the same number of rows")
    n = L.size
    bins = rank_bins(L, n_strata)

    fa0, _ = fano_by_gene(Xc, L, K=K)
    fb0, _ = fano_by_gene(Xc[~is_a], L[~is_a], K=K)

    num_a = np.zeros(Xc.shape[1], dtype=np.float64)
    num_b = np.zeros(Xc.shape[1], dtype=np.float64)
    w_sum = 0.0
    used = 0
    cells_used = 0
    for k in range(n_strata):
        sel_a = (bins == k) & is_a
        sel_b = (bins == k) & (~is_a)
        na, nb = int(sel_a.sum()), int(sel_b.sum())
        if na < min_cells or nb < min_cells:
            continue
        fa, _ = fano_by_gene(Xc[sel_a], L[sel_a], K=K)
        fb, _ = fano_by_gene(Xc[sel_b], L[sel_b], K=K)
        w = float(na + nb)  # pooled weight, identical for both groups (E4)
        num_a += w * fa
        num_b += w * fb
        w_sum += w
        used += 1
        cells_used += na + nb

    if used == 0 or w_sum <= 0:
        return None
    f_a = num_a / w_sum
    f_b = num_b / w_sum
    return StratifiedResult(
        delta_naive=fa0 - fb0,
        delta_strat=f_a - f_b,
        f_a_strat=f_a,
        f_b_strat=f_b,
        j_a=fa0 - f_a,
        j_b=fb0 - f_b,
        coverage=cells_used / float(n),
        strata_used=used,
        n_cells_used=cells_used,
    )


@dataclass
class PermutationResult:
    """Null distribution of ``median_g Delta F`` under exchangeable blocks."""

    observed: float
    p_value: float
    n_assignments: int
    method: str
    null_median: float
    null_q_lo: float
    null_q_hi: float

    def as_dict(self) -> dict:
        return {
            "observed": self.observed,
            "p_value": self.p_value,
            "n_assignments": self.n_assignments,
            "method": self.method,
            "null_median": self.null_median,
            "null_q_lo": self.null_q_lo,
            "null_q_hi": self.null_q_hi,
        }


def _chunk_size(n_genes: int, budget: float = 2.0e6) -> int:
    """Rows per chunk so that one (chunk x gene) matrix stays near 16 MB."""
    return max(16, min(1024, int(budget / max(n_genes, 1))))


def permutation_null_delta(
    stats: SufficientStats,
    is_a_blocks: Sequence[bool],
    gene_mask: Optional[Sequence[bool]] = None,
    n_perm: int = 1000,
    exact_max: int = 20000,
    seed: int = 0,
) -> PermutationResult:
    """Permutation null for the median gene-wise Fano difference.

    Parameters
    ----------
    stats : SufficientStats
        One row per exchangeable block (donor, batch, ...).
    is_a_blocks : sequence of bool
        Group label per block.
    gene_mask : sequence of bool, optional
        Genes entering the statistic. It must be fixed before permutation, and
        the same mask is used for every assignment (conditional inference on the
        gene set).
    n_perm : int
        Monte-Carlo draws when exact enumeration is not feasible.
    exact_max : int
        Enumerate exactly when ``C(n_blocks, n_A) <= exact_max``.
    seed : int

    Returns
    -------
    PermutationResult
    """
    is_a = np.asarray(is_a_blocks, dtype=bool)
    if is_a.size != stats.shape[0]:
        raise ValueError("is_a_blocks must have one entry per block")
    n_a = int(is_a.sum())
    n_b = int(is_a.size - n_a)
    if n_a < 1 or n_b < 1:
        raise ValueError("both groups need at least one block")
    if gene_mask is None:
        mask = np.ones(stats.shape[1], dtype=bool)
    else:
        mask = np.asarray(gene_mask, dtype=bool)
        if mask.size != stats.shape[1]:
            raise ValueError("gene_mask has the wrong length")
    if not np.any(mask):
        raise ValueError("gene_mask selects no genes")

    idx_a = np.where(is_a)[0]
    idx_b = np.where(~is_a)[0]
    pm_a = stats.pool(idx_a)
    pm_b = stats.pool(idx_b)
    observed = float(np.median(pm_a.fano[mask] - pm_b.fano[mask]))
    if not math.isfinite(observed):
        raise ValueError("observed statistic is not finite")

    n_blocks = stats.shape[0]
    total_assignments = math.comb(n_blocks, n_a)
    exact = total_assignments <= exact_max
    if exact:
        assignments = list(combinations(range(n_blocks), n_a))
        method = "exact"
    else:
        if n_perm < 1:
            raise ValueError("n_perm must be positive when enumeration is skipped")
        rng = np.random.default_rng(seed)
        assignments = [tuple(np.sort(rng.choice(n_blocks, n_a, replace=False)))
                       for _ in range(n_perm)]
        method = "monte-carlo"

    s1 = stats.s1[:, mask]
    s2 = stats.s2[:, mask]
    tot1 = s1.sum(axis=0)
    tot2 = s2.sum(axis=0)
    block_n = np.asarray(stats.n, dtype=np.float64)
    total_n = float(block_n.sum())
    g = s1.shape[1]
    chunk = _chunk_size(g)

    null = np.empty(len(assignments), dtype=np.float64)
    for start in range(0, len(assignments), chunk):
        block = assignments[start : start + chunk]
        m = np.zeros((len(block), n_blocks), dtype=np.float64)
        for i, cols in enumerate(block):
            m[i, list(cols)] = 1.0
        a1 = m @ s1
        a2 = m @ s2
        b1 = tot1 - a1
        b2 = tot2 - a2
        # Divide by the number of *cells* in each group, not the number of
        # blocks: blocks (donors) differ in size, and using the block count
        # scales every Fano value by the mean block size.
        na_vec = m @ block_n
        nb_vec = total_n - na_vec
        if np.any(na_vec <= 0) or np.any(nb_vec <= 0):
            raise ValueError("a permutation produced an empty group")
        pa = a1 / na_vec[:, None]
        pb = b1 / nb_vec[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            fa = stats.K * (a2 / na_vec[:, None] - pa * pa) / pa
            fb = stats.K * (b2 / nb_vec[:, None] - pb * pb) / pb
        null[start : start + len(block)] = np.median(fa - fb, axis=1)

    null_ok = null[np.isfinite(null)]
    if null_ok.size == 0:
        raise ValueError("permutation null is entirely non-finite")
    exceed = int(np.sum(np.abs(null_ok) >= abs(observed)))
    p_value = (1.0 + exceed) / (1.0 + float(null_ok.size))
    return PermutationResult(
        observed=observed,
        p_value=float(p_value),
        n_assignments=int(null_ok.size),
        method=method,
        null_median=float(np.median(null_ok)),
        null_q_lo=float(np.quantile(null_ok, 0.025)),
        null_q_hi=float(np.quantile(null_ok, 0.975)),
    )
