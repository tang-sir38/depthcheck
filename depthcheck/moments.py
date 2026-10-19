"""Gene-level moment algebra for normalized single-cell counts.

Model (UMI counts)
------------------
``x_gc | (L_c, p_gc) ~ NB2(mean = L_c p_gc, var = L_c p_gc + alpha_g (L_c p_gc)^2)``
``y_gc = K x_gc / L_c`` with ``K = 10^4`` by default.

Lemma 1 (exact, no independence assumption)::

    F_g = K E[1/L] + K D_g
    D_g = [Cov(p_g, 1/L) + Var(p_g) + alpha_g E[p_g^2]] / pbar_g

Estimable form (Corollary 1.4; exact for every sample, population moments)::

    F = T_depth + T_cov + T_bio
    T_depth = K E[1/L]
    T_cov   = K (E[x/L^2] - pbar E[1/L]) / pbar
    T_bio   = K ((E[x^2/L^2] - E[x/L^2]) - pbar^2) / pbar

The three terms are accumulated from four per-gene sums, which is why the whole
decomposition costs one streaming pass and never materializes a cell x gene
matrix of residuals.

Every ratio that leaves this module is checked with :func:`_finite`; a silent
all-NaN column is the failure mode this package exists to prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

K_DEFAULT = 1e4
"""Normalization constant. Fano values are reported on this scale."""

MIN_MEAN_CP10K = 0.5
"""Default gene filter: ``K * pbar >= 0.5`` in both groups (frozen in E2/E4/E5)."""

MIN_CELLS = 20
"""Default minimum number of cells per group (frozen in E0/E1/E5)."""

MIN_COMMON_GENES = 100
"""Default minimum number of jointly estimable genes (frozen in E0/E5)."""


def _finite(x: float, name: str) -> float:
    """Return ``x`` as float, raising if it is not finite.

    Parameters
    ----------
    x : float
        Value that is about to be written to a result table.
    name : str
        Column or quantity name, used in the error message.
    """
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} is not finite: {v!r}")
    return v


def _as_positive_1d(L: Sequence[float], name: str = "L") -> np.ndarray:
    arr = np.asarray(L, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return arr


def invlib(L: Sequence[float]) -> float:
    """``E[1/L]`` over cells (population mean, ``ddof=0``)."""
    arr = _as_positive_1d(L)
    return _finite(np.mean(1.0 / arr), "E[1/L]")


def delta_invlib(L_a: Sequence[float], L_b: Sequence[float]) -> float:
    """``E[1/L]_A - E[1/L]_B`` with the direction fixed as A minus B."""
    return _finite(invlib(L_a) - invlib(L_b), "delta E[1/L]")


def implied_delta_fano(
    L_a: Sequence[float], L_b: Sequence[float], K: float = K_DEFAULT
) -> float:
    """``K * (E[1/L]_A - E[1/L]_B)`` — the depth-computable part of ``Delta F``.

    This number needs only per-cell total counts. It does not require the
    expression matrix, any model fit, or any assumption about biology.
    """
    if K <= 0:
        raise ValueError("K must be positive")
    return _finite(K * delta_invlib(L_a, L_b), "implied delta Fano")


def effective_depth(L: Sequence[float]) -> float:
    """Harmonic mean depth ``1 / E[1/L]``.

    This is the depth scale that enters the Fano factor. It is not the median:
    the two can move in opposite directions when the left tail of ``L`` differs
    between groups (Proposition 2).
    """
    return _finite(1.0 / invlib(L), "effective depth")


def median_ratio(L_a: Sequence[float], L_b: Sequence[float]) -> float:
    """``median(L_A) / median(L_B)`` — the natural but unreliable diagnostic."""
    a = _as_positive_1d(L_a)
    b = _as_positive_1d(L_b)
    return _finite(float(np.median(a) / np.median(b)), "median depth ratio")


def _safe_ratio(num: float, den: float, name: str, eps: float = 1e-12) -> float:
    """Ratio that is allowed to be undefined only when the denominator is ~0."""
    if abs(den) < eps:
        return float("nan")
    return _finite(num / den, name)


@dataclass
class PooledMoments:
    """Per-gene moments of one pooled group.

    Attributes
    ----------
    pbar : ndarray
        Mean relative abundance, ``E[x/L]``.
    fano : ndarray
        Fano factor on the ``K`` scale, ``K Var(y) / E[y]``.
    t_depth : ndarray
        ``K E[1/L]``, constant across genes.
    t_cov : ndarray
        ``K (E[x/L^2] - pbar E[1/L]) / pbar``.
    t_bio : ndarray
        ``K ((E[x^2/L^2] - E[x/L^2]) - pbar^2) / pbar``.
    n_cells : int
    e1l : float
        ``E[1/L]``.
    K : float
    """

    pbar: np.ndarray
    fano: np.ndarray
    t_depth: np.ndarray
    t_cov: np.ndarray
    t_bio: np.ndarray
    n_cells: int
    e1l: float
    K: float

    def __post_init__(self) -> None:
        n = self.pbar.shape[0]
        for name in ("fano", "t_depth", "t_cov", "t_bio"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} has shape {arr.shape}, expected {(n,)}")
        if self.n_cells <= 0:
            raise ValueError("n_cells must be positive")
        _finite(self.e1l, "E[1/L]")

    def identity_error(self) -> float:
        """Max absolute deviation of ``F - (T_depth + T_cov + T_bio)``.

        Computed only over genes with a strictly positive mean, where all four
        quantities are defined.
        """
        ok = np.isfinite(self.fano) & np.isfinite(self.t_cov) & np.isfinite(self.t_bio)
        if not np.any(ok):
            return float("nan")
        recon = self.t_depth[ok] + self.t_cov[ok] + self.t_bio[ok]
        return float(np.max(np.abs(self.fano[ok] - recon)))

    def delta(self, other: "PooledMoments") -> np.ndarray:
        """``Fano_A - Fano_B`` per gene (A is ``self``)."""
        if self.fano.shape != other.fano.shape:
            raise ValueError("PooledMoments have different gene dimensions")
        return self.fano - other.fano

    def delta_depth(self, other: "PooledMoments") -> float:
        """``K (E[1/L]_A - E[1/L]_B)``."""
        return _finite(self.K * (self.e1l - other.e1l), "delta depth term")


class SufficientStats:
    """Per-block, per-gene sufficient statistics.

    For block ``b`` and gene ``g`` the accumulator stores::

        s1[b, g] = sum_c x / L
        s2[b, g] = sum_c x^2 / L^2
        sc[b, g] = sum_c x / L^2
        n[b]     = number of cells
        si[b]    = sum_c 1 / L

    Blocks are usually donors, but any exchangeable unit works (a donor within a
    cell type, a batch, a single cell). Pooling blocks is a pure sum, so a
    bootstrap or permutation only needs a gather over the block axis.

    Parameters
    ----------
    s1, s2, sc : ndarray, shape (n_blocks, n_genes)
    n : ndarray, shape (n_blocks,)
    si : ndarray, shape (n_blocks,)
    gene_names : sequence of str, optional
    K : float
    """

    __slots__ = ("s1", "s2", "sc", "n", "si", "gene_names", "K")

    def __init__(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        sc: np.ndarray,
        n: np.ndarray,
        si: np.ndarray,
        gene_names: Optional[Sequence[str]] = None,
        K: float = K_DEFAULT,
    ) -> None:
        self.s1 = np.asarray(s1, dtype=np.float64)
        self.s2 = np.asarray(s2, dtype=np.float64)
        self.sc = np.asarray(sc, dtype=np.float64)
        self.n = np.asarray(n, dtype=np.float64)
        self.si = np.asarray(si, dtype=np.float64)
        if self.s1.ndim != 2:
            raise ValueError("s1 must be two-dimensional (blocks x genes)")
        nb, ng = self.s1.shape
        for name, arr in (("s2", self.s2), ("sc", self.sc)):
            if arr.shape != (nb, ng):
                raise ValueError(f"{name} has shape {arr.shape}, expected {(nb, ng)}")
        for name, arr in (("n", self.n), ("si", self.si)):
            if arr.shape != (nb,):
                raise ValueError(f"{name} has shape {arr.shape}, expected {(nb,)}")
        if not np.all(np.isfinite(self.s1)) or not np.all(np.isfinite(self.s2)):
            raise ValueError("sufficient statistics contain non-finite values")
        if not np.all(np.isfinite(self.sc)) or not np.all(np.isfinite(self.si)):
            raise ValueError("sufficient statistics contain non-finite values")
        if np.any(self.n < 0) or np.any(self.si < 0):
            raise ValueError("n and si must be non-negative")
        if gene_names is not None and len(gene_names) != ng:
            raise ValueError("gene_names length does not match the gene axis")
        self.gene_names = None if gene_names is None else list(gene_names)
        self.K = float(K)
        if not (self.K > 0):
            raise ValueError("K must be positive")

    @classmethod
    def zeros(
        cls,
        n_blocks: int,
        n_genes: int,
        gene_names: Optional[Sequence[str]] = None,
        K: float = K_DEFAULT,
    ) -> "SufficientStats":
        return cls(
            np.zeros((n_blocks, n_genes), dtype=np.float64),
            np.zeros((n_blocks, n_genes), dtype=np.float64),
            np.zeros((n_blocks, n_genes), dtype=np.float64),
            np.zeros(n_blocks, dtype=np.float64),
            np.zeros(n_blocks, dtype=np.float64),
            gene_names=gene_names,
            K=K,
        )

    @property
    def shape(self) -> tuple:
        return self.s1.shape

    def add_cells(self, block_ids: np.ndarray, X, L: np.ndarray) -> None:
        """Accumulate one block of cells.

        Parameters
        ----------
        block_ids : ndarray of int, shape (n_rows,)
            Block index for every row of ``X``.
        X : scipy.sparse matrix or ndarray, shape (n_rows, n_genes)
            Raw counts. Sparse input is converted to CSR once per call.
        L : ndarray, shape (n_rows,)
            Per-cell library size (row sums of ``X``).
        """
        from scipy.sparse import csr_matrix, issparse

        Xc = X.tocsr() if issparse(X) else csr_matrix(np.asarray(X))
        L = _as_positive_1d(L)
        block_ids = np.asarray(block_ids, dtype=np.int64)
        if Xc.shape[0] != L.shape[0] or block_ids.shape[0] != L.shape[0]:
            raise ValueError("X, L and block_ids must have the same number of rows")
        if Xc.shape[1] != self.s1.shape[1]:
            raise ValueError("X has a different number of genes than the accumulator")

        inv = 1.0 / L
        inv2 = inv * inv
        X2 = Xc.multiply(Xc)
        for k in np.unique(block_ids):
            sel = block_ids == k
            idx = int(k)
            if idx < 0 or idx >= self.s1.shape[0]:
                raise IndexError(f"block id {idx} out of range")
            self.s1[idx] += np.asarray(Xc[sel].T.dot(inv[sel])).ravel()
            self.s2[idx] += np.asarray(X2[sel].T.dot(inv2[sel])).ravel()
            self.sc[idx] += np.asarray(Xc[sel].T.dot(inv2[sel])).ravel()
            self.n[idx] += float(sel.sum())
            self.si[idx] += float(inv[sel].sum())

    def subset(self, blocks: Iterable[int]) -> "SufficientStats":
        idx = np.asarray(list(blocks), dtype=np.int64)
        return SufficientStats(
            self.s1[idx],
            self.s2[idx],
            self.sc[idx],
            self.n[idx],
            self.si[idx],
            gene_names=self.gene_names,
            K=self.K,
        )

    def pool(self, blocks: Optional[Iterable[int]] = None) -> PooledMoments:
        """Pool the given blocks and return the four per-gene moments."""
        if blocks is None:
            idx = np.arange(self.s1.shape[0])
        else:
            idx = np.asarray(list(blocks), dtype=np.int64)
        if idx.size == 0:
            raise ValueError("cannot pool zero blocks")
        s1 = self.s1[idx].sum(axis=0)
        s2 = self.s2[idx].sum(axis=0)
        sc = self.sc[idx].sum(axis=0)
        n = float(self.n[idx].sum())
        si = float(self.si[idx].sum())
        if n <= 0:
            raise ValueError("pooled block set contains no cells")

        pbar = s1 / n
        e1l = si / n
        with np.errstate(divide="ignore", invalid="ignore"):
            fano = self.K * (s2 / n - pbar * pbar) / pbar
            t_cov = self.K * (sc / n - pbar * e1l) / pbar
            t_bio = self.K * ((s2 / n - sc / n) - pbar * pbar) / pbar
        t_depth = np.full_like(pbar, self.K * e1l)
        return PooledMoments(
            pbar=pbar,
            fano=fano,
            t_depth=t_depth,
            t_cov=t_cov,
            t_bio=t_bio,
            n_cells=int(round(n)),
            e1l=_finite(e1l, "E[1/L]"),
            K=self.K,
        )


def common_gene_mask(
    a: PooledMoments,
    b: PooledMoments,
    min_mean_cp10k: float = MIN_MEAN_CP10K,
    K: Optional[float] = None,
) -> np.ndarray:
    """Genes estimable in both groups.

    A gene enters the contrast when ``K * pbar >= min_mean_cp10k`` in both
    groups and both Fano values are finite. This is the two-sided filter used in
    E2/E4/E5; it is applied before any difference is taken.
    """
    k = a.K if K is None else float(K)
    return (
        (k * a.pbar >= min_mean_cp10k)
        & (k * b.pbar >= min_mean_cp10k)
        & np.isfinite(a.fano)
        & np.isfinite(b.fano)
    )


def ratio_or_nan(num: float, den: float, name: str = "ratio") -> float:
    """Public wrapper around :func:`_safe_ratio` for report tables."""
    return _safe_ratio(num, den, name)
