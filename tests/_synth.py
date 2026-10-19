"""Shared synthetic helpers for the depthcheck test suite.

The generators deliberately share the composition vector ``p`` between groups so
that the only difference between groups is the one the test varies (depth scale
or overdispersion). Drawing a fresh Dirichlet per group would make the two
groups differ in mean expression as well, and the "known" answer would no longer
be known.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def make_params(n_cells: int, n_genes: int, seed: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    L = rng.lognormal(mean=math.log(4000.0), sigma=0.45, size=n_cells)
    p = rng.dirichlet(np.full(n_genes, 0.4))
    p = np.clip(p, 1e-8, None)
    return L, p / p.sum()


def counts(p: np.ndarray, L: np.ndarray, alpha: float, seed: int,
           depth_scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = np.asarray(L, dtype=float) * depth_scale
    mean = L[:, None] * p[None, :]
    if alpha > 0:
        lam = rng.gamma(shape=1.0 / alpha, scale=mean * alpha)
    else:
        lam = mean
    return rng.poisson(lam).astype(np.float64)


def split_donors(n_cells: int, n_donors: int) -> np.ndarray:
    """Deterministic, balanced donor labels."""
    return np.array([f"d{i % n_donors}" for i in range(n_cells)], dtype=object)


def two_group_dataset(
    n_cells: int = 3000,
    n_genes: int = 300,
    depth_scale_a: float = 1.0,
    alpha_a: float = 0.08,
    alpha_b: float = 0.08,
    seed: int = 0,
    n_donors: int = 6,
):
    """Return ``(X, L, is_a, donor)`` for two groups with shared composition."""
    L, p = make_params(n_cells, n_genes, seed=seed)
    xa = counts(p, L, alpha_a, seed=seed + 1, depth_scale=depth_scale_a)
    xb = counts(p, L, alpha_b, seed=seed + 2, depth_scale=1.0)
    X = np.vstack([xa, xb])
    L_all = np.concatenate([L * depth_scale_a, L])
    is_a = np.array([True] * n_cells + [False] * n_cells)
    donor = np.concatenate([split_donors(n_cells, n_donors),
                            np.array([f"b{i % n_donors}" for i in range(n_cells)],
                                     dtype=object)])
    return X, L_all, is_a, donor
