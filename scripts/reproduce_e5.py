#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproduce the E5 application numbers with the published package.

This script is the bridge between the manuscript and the software: it runs the
same 66 (cell type x contrast) units as E5a through ``depthcheck``'s public API
and checks the slope and correlation against the frozen values in
``03_RESULTS/e05_key_numbers.tsv``.

It is deterministic: no bootstrap, no permutation, no timestamps. Two runs must
produce byte-identical output and exit 0.

Usage
-----
    python scripts/reproduce_e5.py --out repro
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depthcheck.io import H5ADCounts
from depthcheck.moments import K_DEFAULT, SufficientStats, common_gene_mask

PROJECT = r"E:\P20_FACTORY\09_FANO_DEPTH"
RAW = os.path.join(PROJECT, "01_RAW_DATA", "GSE267729_mouse_ovary_aging.h5ad")
SAMPLE_MAP = r"E:\P20_FACTORY\08_POI_PRODROME\03_RESULTS\a01_pseudobulk\a01_sample_map.csv"
KEY = os.path.join(PROJECT, "03_RESULTS", "e05_key_numbers.tsv")

GROUPS = ["Y_r", "M_r", "M_ir", "O_a"]
STAGE = {g: i for i, g in enumerate(GROUPS)}
CONTRASTS = [(a, b) for a in GROUPS for b in GROUPS if STAGE[a] > STAGE[b]]
MIN_CELLS = 20
MIN_MEAN = 0.5
MIN_GENES = 100


def fmt(x, sig=10):
    v = float(x)
    if not math.isfinite(v):
        return "NA"
    return "%.*g" % (sig, v)


def ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(y)
    b1, b0 = np.polyfit(x, y, 1)
    yhat = b0 + b1 * x
    s2 = float(np.sum((y - yhat) ** 2) / max(n - 2, 1))
    sxx = float(np.sum((x - x.mean()) ** 2))
    se = math.sqrt(s2 / sxx)
    r = float(np.corrcoef(x, y)[0, 1])
    return dict(n=n, slope=float(b1), intercept=float(b0), r=r, se=se)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="repro")
    ap.add_argument("--block-rows", type=int, default=4096)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    with open(SAMPLE_MAP, "r", encoding="utf-8-sig") as fh:
        donor_group = {row["donor_id"]: row["group"] for row in csv.DictReader(fh)}
    assert len(donor_group) == 17

    with H5ADCounts(RAW, obs_cols=["donor_id", "author_cell_type"]) as ds:
        assert ds.n_cells == 53119 and ds.n_genes == 31729
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
        group_of = np.array([GROUPS.index(donor_group[d]) for d in donors], dtype=np.int64)

        n_units, n_donors, n_groups = len(keep_types), len(donor_levels), len(GROUPS)
        raw_key = (unit_of * n_donors + donor_of) * n_groups + group_of
        block_key, block_index = np.unique(raw_key[keep], return_inverse=True)
        n_blocks = block_key.size
        block_ids = np.full(ds.n_cells, -1, dtype=np.int64)
        block_ids[keep] = block_index
        blk_unit = block_key // (n_groups * n_donors)
        blk_group = block_key % n_groups
        blk_donor = (block_key // n_groups) % n_donors

        stats = SufficientStats.zeros(n_blocks, ds.n_genes,
                                      gene_names=ds.gene_names, K=K_DEFAULT)
        for block in ds.iter_blocks(block_rows=args.block_rows):
            r0 = block.row0
            sel = keep[r0 : r0 + block.L.size]
            if np.any(sel):
                stats.add_cells(block_ids[r0 : r0 + block.L.size][sel],
                                block.X[sel], block.L[sel])

        rows: List[Dict[str, object]] = []
        for ui, unit in enumerate(keep_types):
            for ga, gb in CONTRASTS:
                idx_a = np.where((blk_unit == ui) & (blk_group == STAGE[ga]))[0]
                idx_b = np.where((blk_unit == ui) & (blk_group == STAGE[gb]))[0]
                rec = dict(unit=unit, group_a=ga, group_b=gb,
                           n_cells_a=int(stats.n[idx_a].sum()),
                           n_cells_b=int(stats.n[idx_b].sum()),
                           n_donors_a=int(idx_a.size), n_donors_b=int(idx_b.size))
                if (rec["n_cells_a"] < MIN_CELLS or rec["n_cells_b"] < MIN_CELLS
                        or idx_a.size == 0 or idx_b.size == 0):
                    rec.update(status="not_estimable", n_genes=0,
                               median_dF=float("nan"), K_dE1L=float("nan"),
                               residual=float("nan"))
                    rows.append(rec)
                    continue
                pa = stats.pool(idx_a)
                pb = stats.pool(idx_b)
                mask = common_gene_mask(pa, pb, MIN_MEAN, K=K_DEFAULT)
                if int(mask.sum()) < MIN_GENES:
                    rec.update(status="not_estimable", n_genes=int(mask.sum()),
                               median_dF=float("nan"), K_dE1L=float("nan"),
                               residual=float("nan"))
                    rows.append(rec)
                    continue
                y = float(np.median((pa.fano - pb.fano)[mask]))
                x = float(K_DEFAULT * (pa.e1l - pb.e1l))
                rec.update(status="ok", n_genes=int(mask.sum()), median_dF=y,
                           K_dE1L=x, residual=y - x)
                rows.append(rec)

    fit_rows = [r for r in rows if r["status"] == "ok" and r["unit"] != "Oocytes"]
    fit = ols([r["K_dE1L"] for r in fit_rows], [r["median_dF"] for r in fit_rows])

    with open(os.path.join(out, "repro_e5_units.tsv"), "w", encoding="utf-8",
              newline="") as fh:
        cols = ["unit", "group_a", "group_b", "status", "n_cells_a", "n_cells_b",
                "n_donors_a", "n_donors_b", "n_genes", "median_dF", "K_dE1L", "residual"]
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r[c]) if c in ("n_genes", "median_dF", "K_dE1L",
                                           "residual") else r[c] for c in cols])

    payload = dict(n_units_total=len(rows), n_units_ok=sum(1 for r in rows if r["status"] == "ok"),
                   n_units_fit=len(fit_rows), slope=fit["slope"],
                   intercept=fit["intercept"], r=fit["r"], se=fit["se"])
    with open(os.path.join(out, "repro_e5_fit.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    # ---------------------------------------------------------------- compare
    key = {}
    with open(KEY, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            key[row["key"]] = float(row["value"])
    want_slope = key["slope_all"]
    want_r = key["r_all"]
    want_n = int(key["n_units_primary_fit"])
    d_slope = abs(fit["slope"] - want_slope)
    d_r = abs(fit["r"] - want_r)
    print(f"units: total={len(rows)} ok={payload['n_units_ok']} fit={len(fit_rows)}")
    print(f"slope={fit['slope']:.10f}  (E5 {want_slope:.10f}, |d|={d_slope:.2e})")
    print(f"r    ={fit['r']:.10f}  (E5 {want_r:.10f}, |d|={d_r:.2e})")
    ok = (len(fit_rows) == want_n) and d_slope < 1e-9 and d_r < 1e-9
    print("MATCH" if ok else "MISMATCH")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
