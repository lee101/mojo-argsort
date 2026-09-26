"""Public surface of mojo_argsort: index sorting with Mojo kernels.

Everything here is a thin, honest layer over `src/kernels.mojo`. The shim owns
every array, normalises dtypes, and dispatches to the right kernel; it does no
numeric work of its own.
"""

import numpy as np

from ._lib import (
    INSERTION_LIMIT,
    counting_argsort as _counting_argsort,
    count_descents as _count_descents,
    insertion_argsort as _insertion_argsort,
    radix_argsort as _radix_argsort,
)

__all__ = [
    "argsort",
    "lexsort",
    "counting_argsort",
    "count_descents",
    "is_sorted",
    "SUPPORTED_DTYPES",
]

SUPPORTED_DTYPES = (
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64,
)


def _prepare(x, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype not in [np.dtype(t) for t in SUPPORTED_DTYPES]:
        raise TypeError(
            f"{name}: unsupported dtype {a.dtype}; expected one of "
            f"{[np.dtype(t).name for t in SUPPORTED_DTYPES]}"
        )
    if a.ndim != 1:
        raise ValueError(f"{name}: expected a 1-D array, got shape {a.shape}")
    return np.ascontiguousarray(a)


def argsort(x, *, descending: bool = False, method: str = "auto") -> np.ndarray:
    """Indices that would sort `x`.

    The result is always stable: among equal keys the indices come out in
    ascending input order, matching `numpy.argsort(x, kind="stable")` exactly.

    `method` selects the kernel: "radix" (LSD radix, the default above
    `_lib.INSERTION_LIMIT` elements), "insertion" (comparison sort, the default
    at or below it), or "auto".
    """
    a = _prepare(x, "argsort")
    n = a.size
    if method not in ("auto", "radix", "insertion"):
        raise ValueError(f"unknown method {method!r}")
    if method == "auto":
        method = "insertion" if n <= INSERTION_LIMIT else "radix"
    if method == "insertion":
        return _insertion_argsort(a, descending)
    return _radix_argsort(a, descending)


def lexsort(*keys, descending: bool = False) -> np.ndarray:
    """Lexicographic sort by several keys, last key dominant.

    Same convention as `numpy.lexsort`: `lexsort(b, a)` sorts primarily by `a`.
    Implemented as a chain of stable radix sorts from the least significant key
    outwards, which is the standard least-significant-digit multi-key argument:
    stability carries the earlier keys through as tie-breakers.
    """
    if not keys:
        raise ValueError("lexsort needs at least one key")
    prepared = [_prepare(k, f"lexsort key {i}") for i, k in enumerate(keys)]
    n = prepared[0].size
    for i, k in enumerate(prepared):
        if k.size != n:
            raise ValueError(
                f"lexsort key {i} has length {k.size}, expected {n}"
            )
    idx = np.arange(n, dtype=np.int64)
    for k in prepared:
        idx = idx[_radix_argsort(k[idx], descending)]
    return idx


def counting_argsort(x, buckets: int, *, descending: bool = False) -> np.ndarray:
    """Stable counting sort of 32-bit keys by `key % buckets`."""
    return _counting_argsort(_prepare(x, "counting_argsort"), buckets, descending)


def count_descents(x, *, descending: bool = False) -> int:
    """How many adjacent pairs break the requested order.

    Zero means the input is already sorted in that order. This is the natural
    run statistic a merge or MSD radix sort consults before choosing a
    strategy; `numpy` has no direct equivalent.
    """
    return _count_descents(_prepare(x, "count_descents"), descending)


def is_sorted(x, *, descending: bool = False) -> bool:
    return count_descents(x, descending=descending) == 0
