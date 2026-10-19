"""Backed, block-wise access to H5AD raw counts.

Only three things are needed from an H5AD file: the raw count matrix, the
per-cell library size (the row sum of that matrix), and a few ``obs`` columns.
Loading a 865 MB file into memory is unnecessary and, on a laptop, impossible;
this module therefore streams CSR row blocks and never materializes the full
matrix.

Notes
-----
* ``raw/X`` is preferred over ``X`` when both exist: ``X`` is frequently
  log-normalized, and the decomposition is defined on counts.
* ``L`` is always the row sum of the count matrix, never ``obs/nCount_RNA``.
  Mixing the two breaks the identity (E3-D4).
* CSC input is rejected with an explicit message instead of being silently
  transposed, because a transposed ``L`` would corrupt every downstream number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence

import h5py
import numpy as np
from scipy.sparse import csr_matrix


def _dec(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _decode_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "S":
        return np.array([_dec(v) for v in arr], dtype=object)
    if arr.dtype.kind == "O":
        return np.array([_dec(v) for v in arr], dtype=object)
    if arr.dtype.kind == "U":
        return arr.astype(object)
    return arr


def _read_index(group: h5py.Group) -> np.ndarray:
    for key in ("_index", "index"):
        if key in group:
            return _decode_array(group[key][:])
    raise KeyError(f"no index dataset in group {group.name}")


def read_obs_column(f: h5py.File, column: str) -> np.ndarray:
    """Read one ``obs`` column, decoding the categorical encoding if present."""
    key = f"obs/{column}"
    if key not in f:
        raise KeyError(
            f"obs column {column!r} not found; available: "
            f"{sorted(k for k in f['obs'].keys() if not k.startswith('_'))[:25]}"
        )
    node = f[key]
    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        cats = _decode_array(node["categories"][:])
        codes = np.asarray(node["codes"][:], dtype=np.int64)
        out = np.full(codes.shape, "NA", dtype=object)
        ok = codes >= 0
        out[ok] = cats[codes[ok]]
        return out
    arr = node[:]
    return _decode_array(np.asarray(arr))


@dataclass
class Block:
    """A contiguous row block of the count matrix.

    Attributes
    ----------
    row0 : int
        Global row index of the first cell in the block.
    L : ndarray, shape (n_rows,)
        Per-cell library size, computed as the row sum of the counts.
    X : scipy.sparse.csr_matrix
        Raw counts for these rows.
    """

    row0: int
    L: np.ndarray
    X: csr_matrix


class H5ADCounts:
    """Streaming reader for an H5AD count matrix.

    Use as a context manager::

        with H5ADCounts("data.h5ad", obs_cols=["group", "donor_id"]) as ds:
            for block in ds.iter_blocks():
                ...

    Parameters
    ----------
    path : str
    counts_key : {"auto", "raw/X", "X"}
        ``auto`` prefers ``raw/X``.
    obs_cols : sequence of str, optional
        Columns to read eagerly into :attr:`obs`.
    check_integer : bool
        Check the first block for non-integer values and warn through
        :attr:`counts_are_integer`.
    """

    def __init__(
        self,
        path: str,
        counts_key: str = "auto",
        obs_cols: Optional[Sequence[str]] = None,
        check_integer: bool = True,
    ) -> None:
        self.path = str(path)
        self._f = h5py.File(self.path, "r")
        self.counts_key = self._resolve_counts_key(counts_key)
        node = self._f[self.counts_key]
        self._is_sparse = isinstance(node, h5py.Group)
        if self._is_sparse:
            self.encoding = _dec(node.attrs.get("encoding-type", ""))
            if self.encoding not in ("csr_matrix", "csc_matrix"):
                raise ValueError(f"unsupported sparse encoding {self.encoding!r}")
            if self.encoding == "csc_matrix":
                raise NotImplementedError(
                    "CSC counts are not supported: row streaming would silently "
                    "use the wrong axis. Convert to CSR first, e.g. "
                    "`adata.raw.X = adata.raw.X.tocsr()` and rewrite the file."
                )
            shape = tuple(int(v) for v in node.attrs["shape"])
        else:
            self.encoding = "array"
            shape = tuple(int(v) for v in node.shape)
        self.n_cells, self.n_genes = int(shape[0]), int(shape[1])

        self._var_group = self._resolve_var_group()
        self.gene_names = _read_index(self._var_group)

        self.obs: Dict[str, np.ndarray] = {}
        for col in obs_cols or ():
            self.obs[col] = read_obs_column(self._f, col)

        self.counts_are_integer: Optional[bool] = None
        if check_integer:
            self.counts_are_integer = self._probe_integer()

    # ------------------------------------------------------------------ setup
    def _resolve_counts_key(self, counts_key: str) -> str:
        if counts_key != "auto":
            if counts_key not in self._f:
                raise KeyError(f"counts key {counts_key!r} not found in {self.path}")
            return counts_key
        if "raw" in self._f and "X" in self._f["raw"]:
            return "raw/X"
        if "X" in self._f:
            return "X"
        raise KeyError("no 'raw/X' or 'X' in the H5AD file")

    def _resolve_var_group(self) -> h5py.Group:
        if self.counts_key == "raw/X" and "raw" in self._f and "var" in self._f["raw"]:
            return self._f["raw/var"]
        return self._f["var"]

    def _probe_integer(self) -> bool:
        node = self._f[self.counts_key]
        if self._is_sparse:
            data = node["data"]
            sample = np.asarray(data[: min(len(data), 200_000)], dtype=np.float64)
        else:
            sample = np.asarray(node[: min(self.n_cells, 50), :], dtype=np.float64).ravel()
        if sample.size == 0:
            return True
        return bool(np.all(np.abs(sample - np.round(sample)) < 1e-9))

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return self.n_cells

    def close(self) -> None:
        try:
            self._f.close()
        finally:
            pass

    def __enter__(self) -> "H5ADCounts":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def iter_blocks(self, block_rows: int = 4096) -> Iterator[Block]:
        """Yield CSR row blocks with their library sizes."""
        if block_rows <= 0:
            raise ValueError("block_rows must be positive")
        node = self._f[self.counts_key]
        if self._is_sparse:
            indptr = np.asarray(node["indptr"][:], dtype=np.int64)
            data_ds, idx_ds = node["data"], node["indices"]
            for r0 in range(0, self.n_cells, block_rows):
                r1 = min(r0 + block_rows, self.n_cells)
                a, b = int(indptr[r0]), int(indptr[r1])
                lens = np.diff(indptr[r0 : r1 + 1])
                local_ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
                data = np.asarray(data_ds[a:b], dtype=np.float64)
                indices = np.asarray(idx_ds[a:b], dtype=np.int64)
                X = csr_matrix((data, indices, local_ptr), shape=(r1 - r0, self.n_genes))
                L = np.asarray(X.sum(axis=1)).ravel()
                if not np.all(L > 0):
                    bad = int(np.sum(L <= 0))
                    raise ValueError(f"{bad} cells with zero library size in rows {r0}:{r1}")
                yield Block(row0=r0, L=L, X=X)
        else:
            for r0 in range(0, self.n_cells, block_rows):
                r1 = min(r0 + block_rows, self.n_cells)
                dense = np.asarray(node[r0:r1, :], dtype=np.float64)
                X = csr_matrix(dense)
                L = np.asarray(X.sum(axis=1)).ravel()
                if not np.all(L > 0):
                    bad = int(np.sum(L <= 0))
                    raise ValueError(f"{bad} cells with zero library size in rows {r0}:{r1}")
                yield Block(row0=r0, L=L, X=X)

    def library_sizes(self, block_rows: int = 8192) -> np.ndarray:
        """Per-cell library sizes for the whole file (streaming)."""
        out = np.empty(self.n_cells, dtype=np.float64)
        for block in self.iter_blocks(block_rows=block_rows):
            out[block.row0 : block.row0 + block.L.size] = block.L
        return out

    def read_dense(self, max_cells: int = 20000) -> np.ndarray:
        """Small-matrix escape hatch, mostly for tests and demos."""
        if self.n_cells > max_cells:
            raise ValueError(
                f"refusing to densify {self.n_cells} cells (limit {max_cells})"
            )
        return np.vstack([block.X.toarray() for block in self.iter_blocks()])

    @property
    def var_names(self) -> np.ndarray:
        return self.gene_names


def read_h5ad_counts(
    path: str,
    counts_key: str = "auto",
    obs_cols: Optional[Sequence[str]] = None,
    check_integer: bool = True,
) -> H5ADCounts:
    """Convenience constructor; the returned object owns the file handle."""
    return H5ADCounts(
        path, counts_key=counts_key, obs_cols=obs_cols, check_integer=check_integer
    )
