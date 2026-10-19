"""depthcheck — depth-computable component of the single-cell Fano factor.

Public API
----------
``read_h5ad_counts`` / ``H5ADCounts``
    Backed, block-wise access to raw counts and per-cell library sizes.
``SufficientStats``
    Per-block, per-gene sufficient statistics for F, T_depth, T_cov and T_bio.
``implied_delta_fano`` / ``delta_invlib`` / ``effective_depth``
    Closed-form depth term.
``fano_by_gene`` / ``stratified_delta_fano`` / ``permutation_null_delta``
    Fano estimators and the exact/Monte-Carlo label-permutation null.
``decompose`` / ``verdict_of`` / ``Criteria``
    The pre-registered PASS / FAIL / AMBIGUOUS rule.
``write_report``
    Self-contained one-page HTML report.
"""

from ._version import __version__
from .decompose import (
    AMBIGUOUS,
    CRITERIA,
    FAIL,
    PASS,
    Criteria,
    Decomposition,
    decompose,
    verdict_of,
)
from .fano import (
    StratifiedResult,
    fano_by_gene,
    permutation_null_delta,
    stratified_delta_fano,
)
from .io import Block, H5ADCounts, read_h5ad_counts
from .moments import (
    K_DEFAULT,
    PooledMoments,
    SufficientStats,
    common_gene_mask,
    delta_invlib,
    effective_depth,
    implied_delta_fano,
    invlib,
)
from .report import write_report

__all__ = [
    "__version__",
    "AMBIGUOUS",
    "CRITERIA",
    "FAIL",
    "PASS",
    "Criteria",
    "Decomposition",
    "decompose",
    "verdict_of",
    "StratifiedResult",
    "fano_by_gene",
    "permutation_null_delta",
    "stratified_delta_fano",
    "Block",
    "H5ADCounts",
    "read_h5ad_counts",
    "K_DEFAULT",
    "PooledMoments",
    "SufficientStats",
    "common_gene_mask",
    "delta_invlib",
    "effective_depth",
    "implied_delta_fano",
    "invlib",
    "write_report",
]
