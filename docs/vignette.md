# Vignette: from a public H5AD to the application figure

This walkthrough takes about ten minutes on a laptop, including the 865 MB
download. Every number quoted here was produced by the commands shown.

## 1. Install and run the built-in demo (30 seconds)

```bash
pip install depthcheck
depthcheck demo --out demo
```

```
[demo] depth_only: .../demo/demo_depth_only.h5ad
depthcheck 1.0.0
counts: X  encoding=csr_matrix  cells=3200  genes=400  K=10000
blocks: 10  sufficient statistics ~0.000 GB
stream done: cells=3200  total counts=... 
wrote 1 rows to .../demo/out_depth_only

[demo] noise_only: .../demo/demo_noise_only.h5ad
...
wrote 1 rows to .../demo/out_noise_only
```

The two synthetic datasets differ in exactly one thing:

| dataset | manipulation | observed ΔF | implied K·ΔE[1/L] | residual | verdict |
|---|---|---|---|---|---|
| `depth_only` | A sequenced at 70% depth | 1.183 | 1.201 | −0.017 | **FAIL** |
| `noise_only` | A over-dispersed, depth identical | 6.027 | 0.013 | 6.014 | **PASS** |

Read `demo/out_depth_only/depthcheck_report.html` in a browser. The grey band is
the interval that depth alone predicts; the observed point sits inside it, which
is why the verdict is FAIL.

## 2. What the input needs

* An `.h5ad` file with **raw UMI counts** in `raw/X` (preferred) or `X`.
  Log-normalized values in `X` will produce meaningless Fano factors; the CLI
  checks that the counts are integer-valued and records the result in
  `depthcheck_asserts.txt`.
* One `obs` column with the two group labels.
* Optionally a `cell_type` column (to analyse several units in one pass) and a
  `donor_id` column. **The donor column is required for a verdict**: the
  confidence intervals come from resampling donors, and with one block per group
  the tool reports `point_estimate_only` and `AMBIGUOUS` rather than inventing
  an interval.

Cell-level library sizes are always recomputed as the row sums of the count
matrix. `obs/nCount_RNA` is ignored on purpose: mixing the two definitions
breaks the identity.

## 3. Worked example: mouse ovary ageing (GSE267729)

The dataset is public on CELLxGENE (collection
`460007a7-e893-4fc9-8b84-b52679b4561e`, 865 MB, 53,119 cells, 31,729 genes).
Download the H5AD from the collection page and run one unit:

```bash
depthcheck run \
  --h5ad GSE267729_mouse_ovary_aging.h5ad \
  --group-col group \
  --contrast M_ir:M_r \
  --type-col author_cell_type \
  --type-name "Granulosa cells (Nppc-high)" \
  --donor-col donor_id \
  --stratified \
  --out ovary_nppc
```

Output (`depthcheck_result.tsv`):

| field | value |
|---|---|
| cells A / B | 4,277 / 3,385 |
| donor blocks A / B | 4 / 3 |
| genes in contrast | 2,767 |
| observed ΔF | 0.1617359066 |
| implied K·ΔE[1/L] | 0.1720319955 |
| residual | −0.0102960889 |
| residual 95% CI | −0.061 to +0.040 |
| depth-balanced ΔF (sensitivity) | −0.003512 |
| verdict | **FAIL** |

The depth term reproduces the contrast to within 0.011 Fano units; this unit
cannot be read as a noise difference. That is the intended use of the tool: it
does not tell you whether the biology is real, it tells you whether the number
you would report is identifiable.

## 4. Reproduce the 66-unit application (one pass, ~30 seconds)

`scripts/reproduce_e5.py` runs all 11 cell types × 6 ordered group contrasts from
a single streaming pass and fits the cross-unit regression:

```bash
python scripts/reproduce_e5.py --out repro
```

```
units: total=66 ok=61 fit=60
slope=0.9549254095  (E5 0.9549254095, |d|=2.34e-11)
r    =0.9967693981  (E5 0.9967693981, |d|=7.77e-12)
MATCH
```

The slope is 0.955, not 1. The gap is the second-order term: the residual is
correlated with the depth difference (−0.51) and is present even between donors
of the same group, where there is no biological contrast. Do not report the
slope as one, and do not report it as significantly below one: the
type-cluster bootstrap interval (0.916–1.002) covers one.

## 5. Reading the output

| file | what to look at |
|---|---|
| `depthcheck_result.tsv` | `verdict`, then `observed_dF`, `implied_dF`, `residual` |
| `depthcheck_report.html` | one-page summary with the prediction band |
| `depthcheck_asserts.txt` | every internal check; a single FAIL means the run is not trustworthy |
| `depthcheck_run.json` | parameters and the exact criteria used |

The verdict rule is fixed in code (`depthcheck.decompose`), not chosen per
dataset.

## 6. Common mistakes

1. **Passing normalized `X`.** Check `counts_integer` in the asserts file.
2. **No donor column.** You get point estimates and `AMBIGUOUS`; that is by
   design.
3. **Reading single genes.** `ΔF` is not identifiable per gene. Use the median
   or mean across genes, and never rank genes by `ΔF`.
4. **Treating `--stratified` as a replacement for the closed form.** Depth
   matching is not distribution matching; the two estimators differ by
   `J_A − J_B`, which is not bounded by `H`. Report both, as the report does.
5. **Comparing Fano values across `K`.** `K = 10^4` by default; values scale
   linearly with `K`.
