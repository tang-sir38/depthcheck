"""End-to-end CLI test: the synthetic demo and byte-level determinism."""

from __future__ import annotations

import csv
import os

from depthcheck.cli import main


def _read_verdict(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == 1
    return rows[0]["verdict"]


def test_demo_produces_expected_verdicts(tmp_path, capsys) -> None:
    out = tmp_path / "demo"
    code = main(["demo", "--out", str(out)])
    capsys.readouterr()
    assert code == 0
    assert _read_verdict(str(out / "out_depth_only" / "depthcheck_result.tsv")) == "FAIL"
    assert _read_verdict(str(out / "out_noise_only" / "depthcheck_result.tsv")) == "PASS"
    for name in ("depthcheck_result.tsv", "depthcheck_run.json",
                 "depthcheck_report.html", "depthcheck_summary.html",
                 "depthcheck_asserts.txt"):
        assert (out / "out_noise_only" / name).exists(), name


def test_cli_outputs_are_byte_identical(tmp_path, capsys) -> None:
    demo = tmp_path / "demo"
    assert main(["demo", "--out", str(demo)]) == 0
    h5 = demo / "demo_depth_only.h5ad"
    common = [
        "run", "--h5ad", str(h5), "--group-col", "group", "--contrast", "A:B",
        "--type-col", "cell_type", "--donor-col", "donor_id",
        "--bootstrap", "100", "--permutations", "100", "--stratified",
    ]
    a = tmp_path / "a"
    b = tmp_path / "b"
    assert main(common + ["--out", str(a)]) == 0
    assert main(common + ["--out", str(b)]) == 0
    capsys.readouterr()
    names = sorted(os.listdir(a))
    assert names
    for name in names:
        pa = a / name
        pb = b / name
        if not pa.is_file():
            continue
        assert pa.read_bytes() == pb.read_bytes(), name
