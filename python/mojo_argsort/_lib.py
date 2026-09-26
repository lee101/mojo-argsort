"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-argsort.so"

# key mapping modes, see src/kernels.mojo
MODE_UNSIGNED = 0
MODE_SIGNED = 1
MODE_FLOAT = 2

INSERTION_LIMIT = 64


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.mas_radix_argsort.restype = None
    lib.mas_radix_argsort.argtypes = [ctypes.c_int64] * 11
    lib.mas_insertion_argsort.restype = None
    lib.mas_insertion_argsort.argtypes = [ctypes.c_int64] * 7
    lib.mas_counting_argsort.restype = None
    lib.mas_counting_argsort.argtypes = [ctypes.c_int64] * 9
    lib.mas_count_descents.restype = ctypes.c_int64
    lib.mas_count_descents.argtypes = [ctypes.c_int64] * 5
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


def _widen(keys: np.ndarray) -> np.ndarray:
    """Widen a numeric array without changing the order of its values.

    int8/16/32 -> int64, uint8/16/32 -> uint64, float16/32 -> float64. Every one
    of these is exact, so the resulting ordering is identical to the input's.
    """
    if keys.dtype.itemsize == 8:
        return keys
    if keys.dtype.kind == "f":
        return keys.astype(np.float64)
    if keys.dtype.kind == "u":
        return keys.astype(np.uint64)
    if keys.dtype.kind == "i":
        return keys.astype(np.int64)
    raise TypeError(f"unsupported dtype: {keys.dtype}")


def _mode_of(dtype: np.dtype) -> int:
    if dtype.kind == "u":
        return MODE_UNSIGNED
    if dtype.kind == "i":
        return MODE_SIGNED
    if dtype.kind == "f":
        return MODE_FLOAT
    raise TypeError(f"unsupported dtype: {dtype}")


def radix_argsort(keys: np.ndarray, descending: bool = False) -> np.ndarray:
    """Permutation indices that sort `keys`; stable, integer arithmetic only.

    The kernel reads 32-bit or 64-bit key words, so narrower dtypes are widened
    first. The caller owns the returned array.
    """
    keys = np.ascontiguousarray(keys)
    mode = _mode_of(keys.dtype)
    if keys.dtype.itemsize not in (4, 8):
        keys = _widen(keys)
    n = keys.size
    idx = np.empty(n, dtype=np.int64)
    if n == 0:
        return idx
    ka = np.empty(n, dtype=np.uint64)
    kb = np.empty(n, dtype=np.uint64)
    va = np.empty(n, dtype=np.uint64)
    vb = np.empty(n, dtype=np.uint64)
    hist = np.empty(256, dtype=np.uint32)
    lib.mas_radix_argsort(
        n, _addr(keys), _addr(idx), _addr(ka), _addr(kb), _addr(va), _addr(vb),
        _addr(hist), 8 * keys.dtype.itemsize, mode, 1 if descending else 0,
    )
    return idx


def insertion_argsort(keys: np.ndarray, descending: bool = False) -> np.ndarray:
    """Stable insertion sort of 64-bit key words. Intended for short inputs."""
    keys = _widen(np.ascontiguousarray(keys))
    n = keys.size
    idx = np.empty(n, dtype=np.int64)
    if n == 0:
        return idx
    mode = _mode_of(keys.dtype)
    ka = np.empty(n, dtype=np.uint64)
    va = np.empty(n, dtype=np.uint64)
    lib.mas_insertion_argsort(
        n, _addr(keys), _addr(idx), _addr(ka), _addr(va), mode,
        1 if descending else 0,
    )
    return idx


def counting_argsort(
    keys: np.ndarray, buckets: int, descending: bool = False
) -> np.ndarray:
    """Stable counting sort of 32-bit keys by `key % buckets`."""
    keys = np.ascontiguousarray(keys)
    if buckets <= 0 or (buckets & (buckets - 1)) != 0:
        raise ValueError(f"buckets must be a positive power of two, got {buckets}")
    if keys.dtype.itemsize not in (4, 8):
        raise TypeError(
            f"counting_argsort needs a 32-bit or 64-bit numeric dtype, got "
            f"{keys.dtype}"
        )
    n = keys.size
    idx = np.empty(n, dtype=np.int64)
    if n == 0:
        return idx
    ka = np.empty(n, dtype=np.uint32)
    va = np.empty(n, dtype=np.uint32)
    hist = np.empty(buckets, dtype=np.uint32)
    lib.mas_counting_argsort(
        n, _addr(keys), _addr(idx), _addr(ka), _addr(va), _addr(hist),
        int(buckets), _mode_of(keys.dtype), 1 if descending else 0,
    )
    return idx


def count_descents(keys: np.ndarray, descending: bool = False) -> int:
    """Number of adjacent pairs whose mapped key strictly decreases."""
    keys = np.ascontiguousarray(keys)
    mode = _mode_of(keys.dtype)
    if keys.dtype.itemsize not in (4, 8):
        keys = _widen(keys)
    return int(
        lib.mas_count_descents(
            keys.size, _addr(keys), 8 * keys.dtype.itemsize, mode,
            1 if descending else 0,
        )
    )


widening_for_order = _widen
