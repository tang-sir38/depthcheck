# Release process (maintainers)

`depthcheck/_version.py` is the single source of truth for the version.
`pyproject.toml` reads the same string, and `tests/test_version.py` pins the two
together so that a release cannot go out with a stale version in the paper.

## Cutting a release

```bash
python -m pytest -q                 # fast suite
python -m pytest -q --runslow       # end-to-end regression, needs the H5AD
python -m build                     # writes dist/*.whl and dist/*.tar.gz
```

Then:

1. Tag the commit `vX.Y.Z` and publish a GitHub release from that tag.
2. Deposit `dist/*.whl` and `dist/*.tar.gz` together in one Zenodo record, so the
   DOI resolves to exactly the artefacts that were tested.
3. Put the DOI in `CITATION.cff` and in the README, then rebuild — an archive
   that does not carry its own DOI is not citable.

## Why both files

The wheel is what users install; the sdist is what preserves the source tree
(the wheel omits `tests/`, `docs/` and `scripts/`). Reviewers who want to check
the claims need the sdist.

## Version policy

Uploading is irreversible for a version number. If a file is wrong after a tag
is published, bump the patch version and deposit again; never try to replace the
files of an existing DOI.
