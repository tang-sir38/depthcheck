"""Single source of truth for the package version.

The version is written into the manuscript and into the Zenodo deposit; bump it
here only, and re-run the test suite afterwards (``tests/test_version.py`` pins
the string so a release can never go out with a stale version in the paper).
"""

__version__ = "1.0.0"
