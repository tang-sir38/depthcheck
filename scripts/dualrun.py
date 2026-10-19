#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dual-run harness with the two checks that matter.

A hash comparison alone is not evidence: if a script crashes before writing,
the previous run's files are still on disk and the hashes trivially agree. This
harness therefore

1. moves any previous output directory out of the way *before* running,
2. requires both runs to exit with code 0,
3. requires every expected file to exist, and
4. only then compares SHA-256 digests byte for byte.

Usage
-----
    python scripts/dualrun.py --out repro -- python scripts/reproduce_e5.py --out repro
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(root: str) -> dict:
    files = {}
    for dirpath, _, names in os.walk(root):
        for name in names:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            files[rel] = sha256(full)
    return files


def move_aside(path: str) -> str:
    if not os.path.exists(path):
        return ""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = f"{path}.__prev__.{stamp}"
    n = 1
    while os.path.exists(dest):
        dest = f"{path}.__prev__.{stamp}_{n}"
        n += 1
    shutil.move(path, dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory produced by the command")
    ap.add_argument("--report", default=None)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("no command given after --")

    out = os.path.abspath(args.out)
    report = args.report or os.path.join(os.path.dirname(out), "dualrun_report.json")
    moved = []
    snaps = []
    codes = []
    for run in (1, 2):
        moved.append(move_aside(out))
        os.makedirs(out, exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        codes.append(proc.returncode)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:])
            sys.stderr.write(proc.stderr[-4000:])
            print(f"run {run} exited {proc.returncode}; stopping")
            return 2
        snaps.append(snapshot(out))

    same_names = set(snaps[0]) == set(snaps[1])
    diffs = sorted(k for k in snaps[0] if k not in snaps[1] or snaps[0][k] != snaps[1].get(k))
    payload = {
        "command": " ".join(cmd),
        "exit_codes": codes,
        "files": len(snaps[0]),
        "same_file_set": same_names,
        "n_different": len(diffs),
        "different": diffs,
        "moved_aside": moved,
        "sha256": snaps[0],
    }
    with open(report, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    ok = (codes == [0, 0]) and same_names and not diffs
    print(f"exit codes: {codes}")
    print(f"files compared: {len(snaps[0])}")
    print("byte-identical: " + ("YES" if ok else "NO"))
    if diffs:
        print("differences: " + ", ".join(diffs))
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
