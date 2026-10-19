"""Reproduce the ten reliable mouse ovary cell types from the raw H5AD.

This is the end-to-end regression test: it recomputes the ``M_ir - M_r``
contrast from the 865 MB count matrix with the package's own streaming reader
and compares both the median ``Delta F`` and the number of jointly estimable
genes with the frozen E0 summary table.

Tolerance: the summary was written with ``%.8g``, so a round-trip error of
1e-8 is expected. The test allows 1e-5, which is still three orders of magnitude
tighter than the smallest effect the analysis reports (0.05 Fano units).
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from depthcheck.io import H5ADCounts
from depthcheck.moments import K_DEFAULT, SufficientStats, common_gene_mask

# Fixtures are not distributed with the source (the count matrix is 865 MB).
# Point these two variables at a local copy to enable the test; without them the
# test skips rather than fails.
ROOT = os.environ.get("DEPTHCHECK_ROOT", "")
RAW = os.path.join(ROOT, "01_RAW_DATA", "GSE267729_mouse_ovary_aging.h5ad") if ROOT else ""
E0_SUMMARY = os.path.join(ROOT, "03_RESULTS", "a1_A1_audit_summary.tsv") if ROOT else ""
SAMPLE_MAP = os.environ.get("DEPTHCHECK_SAMPLE_MAP", "")

TOL = 1e-5
GROUPS = ("M_ir", "M_r")


def _read_summary():
    with open(E0_SUMMARY, "r", encoding="utf-8") as fh:
        return {row["type"]: row for row in csv.DictReader(fh, delimiter="\t")}


def _read_sample_map():
    with open(SAMPLE_MAP, "r", encoding="utf-8-sig") as fh:
        return {row["donor_id"]: row["group"] for row in csv.DictReader(fh)}


@pytest.mark.slow
def test_reproduces_e0_ten_types() -> None:
    for path in (RAW, E0_SUMMARY, SAMPLE_MAP):
        if not os.path.exists(path):
            pytest.skip(f"missing fixture: {path}")
    summary = _read_summary()
    donor_group = _read_sample_map()

    with H5ADCounts(RAW, obs_cols=["donor_id", "author_cell_type"]) as ds:
        assert ds.n_cells == 53119
        assert ds.n_genes == 31729
        assert ds.counts_are_integer
        donors = ds.obs["donor_id"]
        types = ds.obs["author_cell_type"]

        keep = np.array([donor_group.get(d) in GROUPS for d in donors])
        keep_types = sorted(set(types[keep].tolist()))
        assert len(keep_types) == 11

        unit_lookup = {t: i for i, t in enumerate(keep_types)}
        donor_levels = sorted(set(donors[keep].tolist()))
        donor_lookup = {d: i for i, d in enumerate(donor_levels)}
        unit_of = np.array([unit_lookup[t] for t in types], dtype=np.int64)
        donor_of = np.array([donor_lookup.get(d, -1) for d in donors], dtype=np.int64)
        group_of = np.array([1 if donor_group.get(d) == "M_ir" else 0 for d in donors],
                            dtype=np.int64)

        n_units, n_donors = len(keep_types), len(donor_levels)
        raw_key = (unit_of * n_donors + donor_of) * 2 + group_of
        block_key, block_index = np.unique(raw_key[keep], return_inverse=True)
        n_blocks = block_key.size
        block_ids = np.full(ds.n_cells, -1, dtype=np.int64)
        block_ids[keep] = block_index
        blk_unit = block_key // (2 * n_donors)
        blk_group = block_key % 2

        expected_blocks = sum(int(summary[t]["ndonnor_M_ir"]) + int(summary[t]["ndonnor_M_r"])
                              for t in keep_types)
        assert n_blocks == expected_blocks, (n_blocks, expected_blocks)

        stats = SufficientStats.zeros(n_blocks, ds.n_genes, K=K_DEFAULT)
        for block in ds.iter_blocks(block_rows=2000):
            r0 = block.row0
            sel = keep[r0 : r0 + block.L.size]
            if np.any(sel):
                stats.add_cells(block_ids[r0 : r0 + block.L.size][sel],
                                block.X[sel], block.L[sel])
        assert abs(stats.n.sum() - keep.sum()) < 1e-6

    max_diff = 0.0
    for ui, unit in enumerate(keep_types):
        row = summary[unit]
        if int(row["reliable"]) != 1:
            continue
        idx_a = np.where((blk_unit == ui) & (blk_group == 1))[0]
        idx_b = np.where((blk_unit == ui) & (blk_group == 0))[0]
        pa = stats.pool(idx_a)
        pb = stats.pool(idx_b)
        mask = common_gene_mask(pa, pb)
        assert int(mask.sum()) == int(row["n_genes_common"]), unit
        observed = float(np.median((pa.fano - pb.fano)[mask]))
        want = float(row["median_dF"])
        diff = abs(observed - want)
        max_diff = max(max_diff, diff)
        assert diff < TOL, f"{unit}: {observed} vs {want}"

    assert max_diff < TOL
