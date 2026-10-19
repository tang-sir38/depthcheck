#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deposit a release on Zenodo and print the DOI.

Requires a personal access token with the ``deposit:write`` and
``deposit:actions`` scopes:

    export ZENODO_TOKEN=...

Example
-------
    python scripts/upload_zenodo.py \
        --file dist/depthcheck-1.0.0.tar.gz \
        --file dist/depthcheck-1.0.0-py3-none-any.whl \
        --metadata .zenodo.json --publish

The script never invents a DOI: if the token is missing it stops with a clear
message. Uploading is done through the deposition bucket, which is the only
method that works for files above the legacy 100 MB endpoint limit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _request(method: str, url: str, token: str, data: bytes = None,
             content_type: str = "application/json") -> dict:
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise SystemExit(f"Zenodo {method} {url} failed: {exc.code}\n{detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", required=True)
    ap.add_argument("--metadata", default=".zenodo.json")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--sandbox", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        print("ZENODO_TOKEN is not set; refusing to create a deposition.", file=sys.stderr)
        print("Set it and re-run. No DOI has been minted.", file=sys.stderr)
        return 2

    base = "https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api"
    dep = _request("POST", f"{base}/deposit/depositions", token, b"{}")
    bucket = dep["links"]["bucket"]
    self_url = dep["links"]["self"]

    for path in args.file:
        if not os.path.exists(path):
            raise SystemExit(f"missing file: {path}")
        name = os.path.basename(path)
        with open(path, "rb") as fh:
            payload = fh.read()
        req = urllib.request.Request(f"{bucket}/{name}", data=payload, method="PUT")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/octet-stream")
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()
        print(f"uploaded {name} ({len(payload)} bytes)")

    with open(args.metadata, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    _request("PUT", self_url, token,
             json.dumps({"metadata": metadata}).encode("utf-8"))

    if args.publish:
        dep = _request("POST", dep["links"]["publish"], token)
    doi = dep.get("doi") or dep.get("metadata", {}).get("doi")
    concept = dep.get("conceptdoi")
    print(f"deposition: {dep.get('links', {}).get('html', '')}")
    print(f"version DOI: {doi}")
    print(f"concept DOI: {concept}")
    if not args.publish:
        print("not published: review the record, then POST the publish link")
    return 0 if doi else 3


if __name__ == "__main__":
    raise SystemExit(main())
