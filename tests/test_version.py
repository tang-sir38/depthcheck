"""The version string is part of the manuscript; keep the two in lockstep."""

from __future__ import annotations

import os
import tomllib

from depthcheck._version import __version__

HERE = os.path.dirname(os.path.abspath(__file__))
PYPROJECT = os.path.join(os.path.dirname(HERE), "pyproject.toml")


def test_version_matches_pyproject() -> None:
    with open(PYPROJECT, "rb") as fh:
        data = tomllib.load(fh)
    assert data["project"]["version"] == __version__


def test_version_is_semver_like() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
