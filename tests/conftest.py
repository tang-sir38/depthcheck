"""Test configuration: path setup and the opt-in slow marker."""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run the tests that need the 865 MB mouse ovary H5AD",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--runslow") or os.environ.get("DEPTHCHECK_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow: pass --runslow or set DEPTHCHECK_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
