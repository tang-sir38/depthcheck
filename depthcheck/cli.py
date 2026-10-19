"""Command line interface and pipeline orchestration.

``depthcheck run`` performs one streaming pass over the count matrix and can
evaluate several (cell type x contrast) units from that pass. Nothing is
materialized beyond the per-block sufficient statistics and the per-cell
library sizes.

Design rules inherited from the project's audit trail
-----------------------------------------------------
* every ratio is checked for finiteness before it is written;
* a non-estimable unit is reported with a reason, never dropped;
* outputs contain no timestamps, so two runs are byte-identical;
* the process exits non-zero if any assertion fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as st

from ._version import __version__
from .decompose import AMBIGUOUS, CRITERIA, CRITERIA_VERSION, Decomposition, decompose
from .fano import permutation_null_delta, rank_bins
from .io import H5ADCounts
from .moments import (
    MIN_CELLS,
    MIN_COMMON_GENES,
    MIN_MEAN_CP10K,
    SufficientStats,
    common_gene_mask,
    effective_depth,
    median_ratio,
)
from .report import write_report, write_summary

RESULT_COLUMNS = [
    "unit", "group_a", "group_b", "status", "reason",
    "n_cells_a", "n_cells_b", "n_donors_a", "n_donors_b", "n_genes",
    "observed_dF", "implied_dF", "residual",
    "observed_ci_lo", "observed_ci_hi",
    "implied_ci_lo", "implied_ci_hi",
    "residual_ci_lo", "residual_ci_hi",
    "H_abs", "ratio_obs_over_implied", "frac_same_sign", "median_abs_dF",
    "e1l_a", "e1l_b", "eff_depth_a", "eff_depth_b",
    "median_depth_a", "median_depth_b", "median_depth_ratio",
    "deltaF_strat", "strat_coverage", "strata_used", "J_a", "J_b", "J_gap",
    "p_exact", "permutation_method", "n_perm_assignments",
    "verdict", "criteria_version",
]

NUMERIC_COLUMNS = [
    "observed_dF", "implied_dF", "residual", "observed_ci_lo", "observed_ci_hi",
    "implied_ci_lo", "implied_ci_hi", "residual_ci_lo", "residual_ci_hi",
    "H_abs", "ratio_obs_over_implied", "frac_same_sign", "median_abs_dF",
    "e1l_a", "e1l_b", "eff_depth_a", "eff_depth_b",
    "median_depth_a", "median_depth_b", "median_depth_ratio",
    "deltaF_strat", "strat_coverage", "J_a", "J_b", "J_gap", "p_exact",
]


def fmt(x, sig: int = 10) -> str:
    """Deterministic numeric formatting.

    Uses the ``%`` operator on purpose: ``"%.10g".format(x)`` returns the
    template unchanged and silently turned 73 key numbers into the literal
    string ``%.10g`` in an earlier experiment (E4 section 6.3).
    """
    if x is None:
        return "NA"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "NA"
    s = "%.*g" % (sig, v)
    if "%" in s:  # pragma: no cover - defensive, cannot happen with %-format
        raise AssertionError("formatting left a template placeholder")
    return s


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in str(text)]
    out = "".join(keep).strip("_")
    return out or "unit"


class Audit:
    """Collects PASS/FAIL assertions; the process exits non-zero on any FAIL."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        if not ok:
            self.failed += 1
        self.lines.append(f"{status}\t{name}\t{detail}")

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(self.lines) + "\n")


def _parse_contrast(text: str) -> Tuple[str, str]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("contrast must be written as GROUP_A:GROUP_B")
    a, b = text.split(":", 1)
    if not a or not b or a == b:
        raise argparse.ArgumentTypeError(f"invalid contrast {text!r}")
    return a, b


def _t_ci(values: Sequence[float], df: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Mean +/- t * SD, the interval used throughout the project (never percentile)."""
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size < 2 or df < 1:
        return (float("nan"), float("nan"))
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return (mean, mean)
    crit = float(st.t.ppf(1 - alpha / 2, df))
    return (mean - crit * sd, mean + crit * sd)


def _bootstrap(
    stats: SufficientStats,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    mask: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    """Cluster bootstrap over blocks (donors) within each group."""
    na, nb = idx_a.size, idx_b.size
    nan = (float("nan"), float("nan"))
    if n_boot < 2 or na < 2 or nb < 2:
        return {"observed": nan, "implied": nan, "residual": nan}
    rng = np.random.default_rng(seed)
    obs, imp, res = [], [], []
    for _ in range(n_boot):
        ia = idx_a[rng.integers(0, na, na)]
        ib = idx_b[rng.integers(0, nb, nb)]
        pa = stats.pool(ia)
        pb = stats.pool(ib)
        med = float(np.median(pa.fano[mask] - pb.fano[mask]))
        x = float(stats.K * (pa.e1l - pb.e1l))
        obs.append(med)
        imp.append(x)
        res.append(med - x)
    df = 2 * (min(na, nb) - 1)
    return {
        "observed": _t_ci(obs, df),
        "implied": _t_ci(imp, df),
        "residual": _t_ci(res, df),
    }


def _memory_guard(n_blocks: int, n_genes: int, limit_gb: float) -> float:
    need = n_blocks * n_genes * 8.0 * 3.0 / 1e9
    if need > limit_gb:
        raise SystemExit(
            f"refusing to allocate {need:.2f} GB of sufficient statistics "
            f"({n_blocks} blocks x {n_genes} genes); raise --max-gb or restrict "
            f"--type-name / --type-col"
        )
    return need


def _empty_row(unit: str, ga: str, gb: str, reason: str) -> Dict[str, object]:
    row: Dict[str, object] = {c: float("nan") for c in NUMERIC_COLUMNS}
    row.update(
        unit=unit, group_a=ga, group_b=gb, status="not_estimable", reason=reason,
        verdict="NA", criteria_version=CRITERIA_VERSION,
        permutation_method="not-run", n_perm_assignments=0,
        n_cells_a=0, n_cells_b=0, n_donors_a=0, n_donors_b=0, n_genes=0,
    )
    return row


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def run_pipeline(args: argparse.Namespace) -> int:
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    audit = Audit()
    log: List[str] = []

    def emit(msg: str) -> None:
        log.append(msg)
        print(msg, flush=True)

    h5_path = os.path.abspath(args.h5ad)
    audit.check("input_exists", os.path.exists(h5_path), h5_path)
    if not os.path.exists(h5_path):
        raise SystemExit(f"input not found: {h5_path}")
    size_mb = os.path.getsize(h5_path) / 1e6

    contrasts = list(args.contrast or [])
    if not contrasts:
        if not (args.group_a and args.group_b):
            raise SystemExit("provide --contrast A:B or --group-a/--group-b")
        contrasts = [(args.group_a, args.group_b)]

    obs_cols = [args.group_col]
    if args.type_col:
        obs_cols.append(args.type_col)
    if args.donor_col:
        obs_cols.append(args.donor_col)

    with H5ADCounts(h5_path, counts_key=args.counts_key, obs_cols=obs_cols) as ds:
        emit(f"depthcheck {__version__}")
        emit(f"input: {h5_path}  {size_mb:.1f} MB")
        emit(f"counts: {ds.counts_key}  encoding={ds.encoding}  "
             f"cells={ds.n_cells}  genes={ds.n_genes}  K={args.k:g}")
        audit.check("counts_integer", bool(ds.counts_are_integer),
                    "raw counts are integer-valued")
        audit.check("cells_positive", ds.n_cells > 0, str(ds.n_cells))
        audit.check("genes_positive", ds.n_genes > 0, str(ds.n_genes))

        groups = np.asarray(ds.obs[args.group_col], dtype=object)
        wanted = sorted({g for pair in contrasts for g in pair})
        present = set(np.unique(groups).tolist())
        missing = [g for g in wanted if g not in present]
        audit.check("groups_present", not missing, f"missing={missing}")
        if missing:
            raise SystemExit(f"group levels not found: {missing}")

        in_contrast = np.isin(groups, wanted)
        group_levels = sorted(present)
        group_lookup = {g: i for i, g in enumerate(group_levels)}
        group_of_cell = np.array([group_lookup[g] for g in groups], dtype=np.int64)

        if args.type_col:
            types_all = np.asarray(ds.obs[args.type_col], dtype=object)
            if args.type_name:
                keep_types = [t for t in args.type_name if t in set(types_all.tolist())]
                missing_types = [t for t in args.type_name if t not in keep_types]
                audit.check("type_names_found", not missing_types,
                            f"missing={missing_types}")
                type_keep = np.isin(types_all, keep_types)
            else:
                type_keep = np.ones(ds.n_cells, dtype=bool)
        else:
            types_all = np.full(ds.n_cells, "ALL", dtype=object)
            types_all = types_all.astype(object)
            type_keep = np.ones(ds.n_cells, dtype=bool)

        keep = in_contrast & type_keep
        audit.check("cells_selected", int(keep.sum()) > 0, str(int(keep.sum())))
        if int(keep.sum()) == 0:
            raise SystemExit("no cells selected")

        units = sorted(set(types_all[keep].tolist()))
        unit_lookup = {u: i for i, u in enumerate(units)}
        unit_of_cell = np.zeros(ds.n_cells, dtype=np.int64)
        for u in units:
            unit_of_cell[types_all == u] = unit_lookup[u]
        emit(f"units: {len(units)}  cells selected: {int(keep.sum())}")

        n_groups = len(group_levels)
        if args.donor_col:
            donors = np.asarray(ds.obs[args.donor_col], dtype=object)
            donor_levels = sorted(set(donors[keep].tolist()))
            donor_lookup = {d: i for i, d in enumerate(donor_levels)}
            donor_of_cell = np.zeros(ds.n_cells, dtype=np.int64)
            for d in donor_levels:
                donor_of_cell[donors == d] = donor_lookup[d]
            n_donors = len(donor_levels)
        else:
            donor_of_cell = np.zeros(ds.n_cells, dtype=np.int64)
            n_donors = 0

        stride_g = n_groups + 1
        stride_d = n_donors + 1
        raw_key = (unit_of_cell * stride_d + donor_of_cell) * stride_g + group_of_cell
        block_key, block_index = np.unique(raw_key[keep], return_inverse=True)
        n_blocks = int(block_key.size)
        block_ids = np.full(ds.n_cells, -1, dtype=np.int64)
        block_ids[keep] = block_index

        blk_group = (block_key % stride_g).astype(np.int64)
        blk_donor = ((block_key // stride_g) % stride_d).astype(np.int64)
        blk_unit = (block_key // (stride_g * stride_d)).astype(np.int64)
        audit.check("blocks_at_least_units", n_blocks >= len(units), str(n_blocks))

        need_gb = _memory_guard(n_blocks, ds.n_genes, args.max_gb)
        emit(f"blocks: {n_blocks}  sufficient statistics ~{need_gb:.3f} GB")

        stats = SufficientStats.zeros(n_blocks, ds.n_genes,
                                      gene_names=ds.gene_names, K=args.k)
        L_all = np.empty(ds.n_cells, dtype=np.float64)
        sum_L = 0.0
        sum_counts = 0.0
        for block in ds.iter_blocks(block_rows=args.block_rows):
            r0 = block.row0
            r1 = r0 + block.L.size
            L_all[r0:r1] = block.L
            sum_L += float(block.L.sum())
            sum_counts += float(block.X.sum())
            sel = keep[r0:r1]
            if np.any(sel):
                stats.add_cells(block_ids[r0:r1][sel], block.X[sel], block.L[sel])

        audit.check("all_cells_accounted",
                    abs(stats.n.sum() - float(keep.sum())) < 1e-6,
                    f"{stats.n.sum():.0f} vs {keep.sum()}")
        audit.check("L_equals_count_sum", abs(sum_L - sum_counts) < 0.5,
                    f"{sum_L:.1f} vs {sum_counts:.1f}")
        emit(f"stream done: cells={int(stats.n.sum())}  total counts={sum_counts:.0f}")

        strat_stats = None
        strat_bins = None
        if args.stratified:
            # Second pass: per (unit, group, depth stratum) sufficient statistics.
            # Boundaries are fixed from the first pass, so they cannot adapt to
            # the expression values.
            strat_bins = np.full(ds.n_cells, -1, dtype=np.int64)
            for ui in range(len(units)):
                sel_u = keep & (unit_of_cell == ui)
                if int(sel_u.sum()) < args.strata * args.min_cells:
                    continue
                strat_bins[sel_u] = rank_bins(L_all[sel_u], args.strata)
            n_sblocks = len(units) * (n_groups + 1) * (args.strata + 1)
            strat_stats = SufficientStats.zeros(n_sblocks, ds.n_genes,
                                                gene_names=ds.gene_names, K=args.k)
            for block in ds.iter_blocks(block_rows=args.block_rows):
                r0 = block.row0
                idx = np.arange(r0, r0 + block.L.size)
                m = keep[idx] & (strat_bins[idx] >= 0)
                if not np.any(m):
                    continue
                skey = ((unit_of_cell[idx][m] * (n_groups + 1)
                         + group_of_cell[idx][m]) * (args.strata + 1)
                        + strat_bins[idx][m]).astype(np.int64)
                strat_stats.add_cells(skey, block.X[m], block.L[m])
            emit("second pass done (stratified estimator)")

        rows: List[Dict[str, object]] = []
        gene_dump: List[Dict[str, object]] = []
        single_detail: Optional[Decomposition] = None
        single_meta: Dict[str, object] = {}

        for ui, unit in enumerate(units):
            unit_cells = keep & (unit_of_cell == ui)
            for ci, (ga, gb) in enumerate(contrasts):
                gi_a, gi_b = group_lookup[ga], group_lookup[gb]
                idx_a = np.where((blk_unit == ui) & (blk_group == gi_a))[0]
                idx_b = np.where((blk_unit == ui) & (blk_group == gi_b))[0]
                cell_a = unit_cells & (group_of_cell == gi_a)
                cell_b = unit_cells & (group_of_cell == gi_b)
                seed = int(args.seed) + 1000 * ui + 7 * ci

                rec = _empty_row(unit, ga, gb, "")
                rec.update(
                    n_cells_a=int(cell_a.sum()), n_cells_b=int(cell_b.sum()),
                    n_donors_a=int(idx_a.size) if args.donor_col else 1,
                    n_donors_b=int(idx_b.size) if args.donor_col else 1,
                )
                if rec["n_cells_a"] < args.min_cells or rec["n_cells_b"] < args.min_cells:
                    rec["reason"] = f"n_cells_below_{args.min_cells}"
                    rows.append(rec)
                    continue
                if idx_a.size == 0 or idx_b.size == 0:
                    rec["reason"] = "group_missing_for_unit"
                    rows.append(rec)
                    continue

                pa = stats.pool(idx_a)
                pb = stats.pool(idx_b)
                mask = common_gene_mask(pa, pb, args.min_mean, K=args.k)
                n_genes = int(mask.sum())
                rec["n_genes"] = n_genes
                if n_genes < args.min_genes:
                    rec["reason"] = f"n_common_genes_below_{args.min_genes}"
                    rows.append(rec)
                    continue

                d_f = pa.fano[mask] - pb.fano[mask]
                observed = float(np.median(d_f))
                implied = float(args.k * (pa.e1l - pb.e1l))
                residual = observed - implied
                frac_same = float(np.mean(np.sign(d_f) == np.sign(implied))) \
                    if abs(implied) > 0 else float("nan")
                rec.update(
                    observed_dF=observed, implied_dF=implied, residual=residual,
                    H_abs=abs(implied),
                    frac_same_sign=frac_same,
                    median_abs_dF=float(np.median(np.abs(d_f))),
                    e1l_a=pa.e1l, e1l_b=pb.e1l,
                    eff_depth_a=effective_depth(L_all[cell_a]),
                    eff_depth_b=effective_depth(L_all[cell_b]),
                    median_depth_a=float(np.median(L_all[cell_a])),
                    median_depth_b=float(np.median(L_all[cell_b])),
                    median_depth_ratio=median_ratio(L_all[cell_a], L_all[cell_b]),
                    ratio_obs_over_implied=(observed / implied) if abs(implied) > 1e-12
                    else float("nan"),
                )

                err_a = pa.identity_error() / max(1.0, abs(float(np.median(pa.fano[mask]))))
                err_b = pb.identity_error() / max(1.0, abs(float(np.median(pb.fano[mask]))))
                audit.check(f"identity_a_{_slug(unit)}_{ga}_{gb}", err_a < 1e-9,
                            f"rel={err_a:.2e}")
                audit.check(f"identity_b_{_slug(unit)}_{ga}_{gb}", err_b < 1e-9,
                            f"rel={err_b:.2e}")

                boot = _bootstrap(stats, idx_a, idx_b, mask, args.bootstrap, seed)
                rec.update(
                    observed_ci_lo=boot["observed"][0], observed_ci_hi=boot["observed"][1],
                    implied_ci_lo=boot["implied"][0], implied_ci_hi=boot["implied"][1],
                    residual_ci_lo=boot["residual"][0], residual_ci_hi=boot["residual"][1],
                )

                if args.permutations and args.permutations > 0:
                    sub_idx = np.concatenate([idx_a, idx_b])
                    sub = stats.subset(sub_idx)
                    is_a_blocks = np.array([True] * idx_a.size + [False] * idx_b.size)
                    perm = permutation_null_delta(
                        sub, is_a_blocks, gene_mask=mask,
                        n_perm=int(args.permutations),
                        exact_max=int(args.exact_max), seed=seed,
                    )
                    audit.check(f"perm_matches_observed_{_slug(unit)}_{ga}_{gb}",
                                abs(perm.observed - observed) < 1e-9,
                                f"{perm.observed:.12g} vs {observed:.12g}")
                    rec.update(p_exact=perm.p_value,
                               permutation_method=perm.method,
                               n_perm_assignments=perm.n_assignments)

                if args.stratified and strat_stats is not None and strat_bins is not None:
                    fa_num = np.zeros(ds.n_genes)
                    fb_num = np.zeros(ds.n_genes)
                    w_sum = 0.0
                    cells_used = 0
                    used = 0
                    for k in range(args.strata):
                        sk_a = (ui * (n_groups + 1) + gi_a) * (args.strata + 1) + k
                        sk_b = (ui * (n_groups + 1) + gi_b) * (args.strata + 1) + k
                        na = int(strat_stats.n[sk_a])
                        nb = int(strat_stats.n[sk_b])
                        if na < args.min_cells or nb < args.min_cells:
                            continue
                        p_a = strat_stats.pool([sk_a])
                        p_b = strat_stats.pool([sk_b])
                        w = float(na + nb)
                        fa_num += w * p_a.fano
                        fb_num += w * p_b.fano
                        w_sum += w
                        cells_used += na + nb
                        used += 1
                    if used > 0 and w_sum > 0:
                        f_a_strat = fa_num / w_sum
                        f_b_strat = fb_num / w_sum
                        rec.update(
                            deltaF_strat=float(np.median(f_a_strat[mask] - f_b_strat[mask])),
                            strat_coverage=cells_used / float(rec["n_cells_a"] + rec["n_cells_b"]),
                            strata_used=used,
                            J_a=float(np.median(pa.fano[mask] - f_a_strat[mask])),
                            J_b=float(np.median(pb.fano[mask] - f_b_strat[mask])),
                        )
                        rec["J_gap"] = rec["J_a"] - rec["J_b"]

                ci_ok = all(math.isfinite(rec[k]) for k in
                            ("residual_ci_lo", "residual_ci_hi", "implied_ci_lo",
                             "implied_ci_hi"))
                if not ci_ok:
                    rec["status"] = "point_estimate_only"
                    rec["reason"] = "ci_undefined_need_at_least_two_donors_per_group"
                    rec["verdict"] = AMBIGUOUS
                else:
                    dec = decompose(
                        observed=observed, implied=implied,
                        implied_ci=(rec["implied_ci_lo"], rec["implied_ci_hi"]),
                        residual_ci=(rec["residual_ci_lo"], rec["residual_ci_hi"]),
                        observed_ci=(rec["observed_ci_lo"], rec["observed_ci_hi"]),
                        p_exact=rec["p_exact"],
                        permutation_method=str(rec["permutation_method"]),
                        n_assignments=int(rec["n_perm_assignments"]),
                        context={"H_abs": abs(implied), "n_genes": n_genes},
                    )
                    rec["status"] = "ok"
                    rec["reason"] = ""
                    rec["verdict"] = dec.verdict
                    if single_detail is None:
                        single_detail = dec
                        single_meta = {"unit": f"{unit} | {ga} vs {gb}"}

                undefined_by_design = {"ratio_obs_over_implied", "frac_same_sign"}
                for c in NUMERIC_COLUMNS:
                    v = rec[c]
                    if isinstance(v, float) and not math.isfinite(v):
                        allowed = (c in undefined_by_design
                                   and abs(float(rec["implied_dF"])) < 1e-12)
                        if rec["status"] == "ok" and not allowed:
                            raise AssertionError(
                                f"non-finite result column {c} for ok unit {unit}/{ga}-{gb}")
                        rec[c] = float("nan")
                rows.append(rec)

                if args.genes and len(units) * len(contrasts) == 1:
                    for g in np.where(mask)[0]:
                        gene_dump.append(
                            dict(gene=str(ds.gene_names[g]),
                                 fano_a=float(pa.fano[g]), fano_b=float(pb.fano[g]),
                                 delta_fano=float(pa.fano[g] - pb.fano[g]),
                                 pbar_a=float(pa.pbar[g]), pbar_b=float(pb.pbar[g]))
                        )

        n_ok = sum(1 for r in rows if r["status"] == "ok")
        audit.check("at_least_one_unit_ok", n_ok > 0, f"{n_ok}/{len(rows)}")
        optional = {"ratio_obs_over_implied", "frac_same_sign"}
        for c in NUMERIC_COLUMNS:
            if c in optional:
                required = [r[c] for r in rows
                            if r["status"] == "ok" and abs(float(r["implied_dF"])) > 1e-12]
                undefined = [r[c] for r in rows
                             if r["status"] == "ok" and abs(float(r["implied_dF"])) <= 1e-12]
                audit.check(f"ok_rows_finite_{c}",
                            all(math.isfinite(float(v)) for v in required),
                            f"n={len(required)}")
                audit.check(f"undefined_when_implied_zero_{c}",
                            all(not math.isfinite(float(v)) for v in undefined),
                            f"n={len(undefined)}")
                continue
            vals = [r[c] for r in rows if r["status"] == "ok"]
            audit.check(f"ok_rows_finite_{c}",
                        all(math.isfinite(float(v)) for v in vals), f"n={len(vals)}")
        audit.check("no_percent_in_outputs",
                    all("%" not in fmt(r[c]) for r in rows for c in NUMERIC_COLUMNS), "")

        # ---------------------------------------------------------------- write
        result_path = os.path.join(out_dir, "depthcheck_result.tsv")
        with open(result_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv_writer(fh)
            writer.writerow(RESULT_COLUMNS)
            for r in rows:
                writer.writerow([fmt(r[c]) if c in NUMERIC_COLUMNS else str(r[c])
                                 for c in RESULT_COLUMNS])

        if gene_dump:
            gpath = os.path.join(out_dir, "depthcheck_genes.tsv")
            with open(gpath, "w", encoding="utf-8", newline="") as fh:
                writer = csv_writer(fh)
                writer.writerow(["gene", "fano_a", "fano_b", "delta_fano",
                                 "pbar_a", "pbar_b"])
                for row in gene_dump:
                    writer.writerow([row["gene"], fmt(row["fano_a"]), fmt(row["fano_b"]),
                                     fmt(row["delta_fano"]), fmt(row["pbar_a"]),
                                     fmt(row["pbar_b"])])

        meta = {
            "input": os.path.basename(h5_path),
            "input_bytes": int(os.path.getsize(h5_path)),
            "counts_key": ds.counts_key,
            "n_cells_file": int(ds.n_cells),
            "n_genes_file": int(ds.n_genes),
            "group_col": args.group_col,
            "type_col": args.type_col or "",
            "donor_col": args.donor_col or "",
            "K": float(args.k),
            "strata": int(args.strata) if args.stratified else 0,
            "bootstrap": int(args.bootstrap),
            "permutations": int(args.permutations),
            "exact_max": int(args.exact_max),
            "seed": int(args.seed),
            "version": __version__,
            "criteria_version": CRITERIA_VERSION,
        }
        run_json = {
            "tool": "depthcheck",
            "version": __version__,
            "criteria_version": CRITERIA_VERSION,
            "criteria": CRITERIA.as_dict(),
            "meta": meta,
            "rows": rows,
        }
        with open(os.path.join(out_dir, "depthcheck_run.json"), "w",
                  encoding="utf-8", newline="\n") as fh:
            json.dump(_json_safe(run_json), fh, indent=2, sort_keys=True,
                      ensure_ascii=False)
            fh.write("\n")

        report_meta = dict(meta)
        report_meta["subtitle"] = "Depth decomposition of a normalized Fano contrast"
        if single_detail is not None:
            report_meta.update(single_meta)
            write_report(single_detail, os.path.join(out_dir, "depthcheck_report.html"),
                         report_meta)
        write_summary(rows, os.path.join(out_dir, "depthcheck_summary.html"),
                      report_meta, n_ok=n_ok)

        audit.write(os.path.join(out_dir, "depthcheck_asserts.txt"))
        with open(os.path.join(out_dir, "depthcheck_log.txt"), "w",
                  encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(log) + "\n")

    print(f"wrote {len(rows)} rows to {out_dir}")
    return 0 if audit.failed == 0 else 2


def csv_writer(fh):
    import csv
    return csv.writer(fh, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


# ---------------------------------------------------------------------- demo
def _write_demo_h5ad(path: str, X, groups, donors, types, gene_names) -> None:
    """Write a minimal CSR H5AD that both depthcheck and anndata can read."""
    import h5py
    from scipy.sparse import csr_matrix

    X = csr_matrix(X)
    with h5py.File(path, "w") as f:
        f.attrs["h5ad-version"] = "0.8.0"
        g = f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = np.array(X.shape, dtype=np.int64)
        g.attrs["dtype"] = "float32"
        g.create_dataset("data", data=X.data.astype(np.float32))
        g.create_dataset("indices", data=X.indices.astype(np.int32))
        g.create_dataset("indptr", data=X.indptr.astype(np.int32))
        obs = f.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["encoding-version"] = "0.1.0"
        obs.attrs["_index"] = "index"
        obs.create_dataset("_index", data=np.array([f"cell_{i}" for i in range(X.shape[0])],
                                                   dtype=object).astype("S"))
        for name, values in (("group", groups), ("donor_id", donors),
                             ("cell_type", types)):
            uniq = sorted(set(values))
            codes = np.array([uniq.index(v) for v in values], dtype=np.int32)
            node = obs.create_group(name)
            node.attrs["encoding-type"] = "categorical"
            node.attrs["encoding-version"] = "0.2.0"
            node.create_dataset("categories", data=np.array(uniq, dtype=object).astype("S"))
            node.create_dataset("codes", data=codes)
        var = f.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["encoding-version"] = "0.1.0"
        var.attrs["_index"] = "index"
        var.create_dataset("_index", data=np.array(gene_names, dtype=object).astype("S"))


def _simulate(n_cells: int, n_genes: int, depth_scale: float, alpha: float,
              seed: int, p: Optional[np.ndarray] = None,
              L: Optional[np.ndarray] = None) -> np.ndarray:
    """Negative-binomial counts with a shared composition across groups.

    ``p`` and ``L`` can be supplied so that two groups differ *only* in the
    intended factor (depth scale or overdispersion). Drawing a fresh Dirichlet
    for each group would make the two groups differ in mean composition as well,
    which is not what the demo is meant to show.
    """
    rng = np.random.default_rng(seed)
    if L is None:
        L = rng.lognormal(mean=math.log(4000.0), sigma=0.45, size=n_cells)
    L = np.asarray(L, dtype=float) * depth_scale
    if p is None:
        p = rng.dirichlet(np.full(n_genes, 0.35))
    p = np.clip(np.asarray(p, dtype=float), 1e-7, None)
    p = p / p.sum()
    mean = L[:, None] * p[None, :]
    if alpha > 0:
        shape = 1.0 / alpha
        lam = rng.gamma(shape=shape, scale=mean * alpha)
    else:
        lam = mean
    return rng.poisson(lam).astype(np.float64)


def cmd_demo(args: argparse.Namespace) -> int:
    """Build two synthetic datasets with known ground truth and run both."""
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    n_genes, n_cells = 400, 1600
    genes = [f"G{i:04d}" for i in range(n_genes)]

    rng = np.random.default_rng(7)
    L_shared = rng.lognormal(mean=math.log(4000.0), sigma=0.45, size=n_cells)
    p_shared = rng.dirichlet(np.full(n_genes, 0.35))
    p_shared = p_shared / p_shared.sum()

    # Depth only: identical composition and overdispersion, A shallower by 30%.
    # Expected verdict FAIL: the depth term alone reproduces the contrast.
    xa = _simulate(n_cells, n_genes, 0.70, 0.08, seed=11, p=p_shared, L=L_shared)
    xb = _simulate(n_cells, n_genes, 1.00, 0.08, seed=12, p=p_shared, L=L_shared)
    X = np.vstack([xa, xb])
    groups = ["A"] * n_cells + ["B"] * n_cells
    donors = [f"A{i % 5}" for i in range(n_cells)] + [f"B{i % 5}" for i in range(n_cells)]
    types = ["synthetic"] * (2 * n_cells)
    p1 = os.path.join(out, "demo_depth_only.h5ad")
    _write_demo_h5ad(p1, X, groups, donors, types, genes)

    # Noise only: identical composition and identical library sizes, A
    # over-dispersed. Expected verdict PASS.
    xa2 = _simulate(n_cells, n_genes, 1.00, 0.45, seed=21, p=p_shared, L=L_shared)
    xb2 = _simulate(n_cells, n_genes, 1.00, 0.05, seed=22, p=p_shared, L=L_shared)
    X2 = np.vstack([xa2, xb2])
    p2 = os.path.join(out, "demo_noise_only.h5ad")
    _write_demo_h5ad(p2, X2, groups, donors, types, genes)

    status = 0
    for path, label in ((p1, "depth_only"), (p2, "noise_only")):
        sub = argparse.Namespace(
            h5ad=path,
            out=os.path.join(out, f"out_{label}"),
            counts_key="auto",
            group_col="group",
            contrast=[("A", "B")],
            group_a=None,
            group_b=None,
            type_col="cell_type",
            type_name=None,
            donor_col="donor_id",
            min_cells=MIN_CELLS,
            min_mean=MIN_MEAN_CP10K,
            min_genes=MIN_COMMON_GENES,
            strata=5,
            stratified=True,
            bootstrap=200,
            permutations=200,
            exact_max=20000,
            seed=20261018,
            k=1e4,
            block_rows=4096,
            max_gb=1.2,
            genes=False,
        )
        print(f"\n[demo] {label}: {path}")
        status |= run_pipeline(sub)
    return status


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depthcheck",
        description="Decompose the Fano factor of normalized single-cell counts "
                    "into a depth-computable term and a gene-specific residual.",
    )
    parser.add_argument("--version", action="version", version=f"depthcheck {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="analyse one H5AD file")
    run.add_argument("--h5ad", required=True)
    run.add_argument("--counts-key", default="auto",
                     choices=["auto", "raw/X", "X"])
    run.add_argument("--group-col", required=True)
    run.add_argument("--contrast", action="append", type=_parse_contrast, default=None,
                     metavar="A:B", help="repeatable, e.g. --contrast old:young")
    run.add_argument("--group-a", default=None)
    run.add_argument("--group-b", default=None)
    run.add_argument("--type-col", default=None)
    run.add_argument("--type-name", action="append", default=None)
    run.add_argument("--donor-col", default=None)
    run.add_argument("--min-cells", type=int, default=MIN_CELLS)
    run.add_argument("--min-mean", type=float, default=MIN_MEAN_CP10K)
    run.add_argument("--min-genes", type=int, default=MIN_COMMON_GENES)
    run.add_argument("--strata", type=int, default=5)
    run.add_argument("--stratified", action="store_true",
                     help="second pass; adds the depth-balanced estimator")
    run.add_argument("--bootstrap", type=int, default=200)
    run.add_argument("--permutations", type=int, default=200)
    run.add_argument("--exact-max", type=int, default=20000)
    run.add_argument("--seed", type=int, default=20261018)
    run.add_argument("--k", type=float, default=1e4)
    run.add_argument("--block-rows", type=int, default=4096)
    run.add_argument("--max-gb", type=float, default=1.2)
    run.add_argument("--genes", action="store_true",
                     help="write a per-gene table (single unit only)")
    run.add_argument("--out", default="depthcheck_out")
    run.set_defaults(func=run_pipeline)

    demo = sub.add_parser("demo", help="synthetic end-to-end demonstration")
    demo.add_argument("--out", default="depthcheck_demo")
    demo.set_defaults(func=cmd_demo)

    ver = sub.add_parser("version", help="print the version")
    ver.set_defaults(func=lambda a: print(__version__) or 0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
