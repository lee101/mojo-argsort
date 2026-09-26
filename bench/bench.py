"""Correctness-gated benchmark for mojo-argsort.

Every case checks its result against NumPy (or an exact Python reference)
before timing, so a broken kernel shows up as a correctness failure rather than
a suspiciously good number. The baselines are NumPy's fastest formulations:
`kind="stable"` for the stable integer argsort, `np.lexsort` for the multi-key
sort, and `np.count_nonzero(np.diff(...))` for the descent count.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_argsort as ma  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_int32(n: int = 1 << 20):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1 << 20, size=n).astype(np.int32)
    got = ma.argsort(x)
    assert np.array_equal(got, np.argsort(x, kind="stable")), "int32 mismatch"
    return (
        f"argsort int32 n={n}",
        _time(lambda: np.argsort(x, kind="stable")),
        _time(lambda: ma.argsort(x)),
    )


def bench_int32_quicksort(n: int = 1 << 20):
    """NumPy's default introsort, which is not stable, for context."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1 << 20, size=n).astype(np.int32)
    ma.argsort(x)
    np.argsort(x)
    return (
        f"argsort int32 (quicksort ref)",
        _time(lambda: np.argsort(x)),
        _time(lambda: ma.argsort(x)),
    )


def bench_int64(n: int = 1 << 20):
    rng = np.random.default_rng(1)
    x = rng.integers(-(1 << 40), 1 << 40, size=n).astype(np.int64)
    got = ma.argsort(x)
    assert np.array_equal(got, np.argsort(x, kind="stable")), "int64 mismatch"
    return (
        f"argsort int64 n={n}",
        _time(lambda: np.argsort(x, kind="stable")),
        _time(lambda: ma.argsort(x)),
    )


def bench_float64(n: int = 1 << 20):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(n)
    got = ma.argsort(x)
    assert np.array_equal(got, np.argsort(x, kind="stable")), "float64 mismatch"
    return (
        f"argsort float64 n={n}",
        _time(lambda: np.argsort(x, kind="stable")),
        _time(lambda: ma.argsort(x)),
    )


def bench_descending(n: int = 1 << 20):
    rng = np.random.default_rng(3)
    x = rng.integers(0, 1 << 20, size=n).astype(np.int64)
    got = ma.argsort(x, descending=True)
    assert np.array_equal(got, ma.argsort(-x)), "descending must match negated"
    return (
        f"argsort int64 descending n={n}",
        _time(lambda: ma.argsort(-x)),
        _time(lambda: ma.argsort(x, descending=True)),
    )


def bench_small(n: int = 32, repeats: int = 200_000):
    rng = np.random.default_rng(4)
    x = rng.integers(0, 100, size=n)
    got = ma.argsort(x)
    assert np.array_equal(got, np.argsort(x, kind="stable")), "small mismatch"

    def run_mine():
        for _ in range(repeats):
            ma.argsort(x)

    def run_numpy():
        for _ in range(repeats):
            np.argsort(x, kind="stable")

    return f"argsort n={n} x{repeats}", _time(run_numpy, 3), _time(run_mine, 3)


def bench_counting(n: int = 1 << 20, buckets: int = 4096):
    rng = np.random.default_rng(5)
    x = rng.integers(0, 1 << 24, size=n).astype(np.int32)
    got = ma.counting_argsort(x, buckets)
    values = (x[got] % buckets).astype(np.int64)
    assert np.all(values[1:] >= values[:-1]), "counting sort not ordered"
    return (
        f"counting_argsort n={n} b={buckets}",
        _time(lambda: np.argsort(x, kind="stable")),
        _time(lambda: ma.counting_argsort(x, buckets)),
    )


def bench_lexsort(n: int = 1 << 18):
    rng = np.random.default_rng(6)
    a = rng.integers(0, 1 << 16, size=n)
    b = rng.integers(0, 1 << 8, size=n)
    got = ma.lexsort(b, a)
    assert np.array_equal(got, np.lexsort((b, a))), "lexsort mismatch"
    return (
        f"lexsort 2 keys n={n}",
        _time(lambda: np.lexsort((b, a)), 3),
        _time(lambda: ma.lexsort(b, a), 3),
    )


def bench_descents(n: int = 1 << 22):
    rng = np.random.default_rng(7)
    x = np.ascontiguousarray(rng.integers(0, 1 << 20, size=n))
    got = ma.count_descents(x)
    assert got == int(np.count_nonzero(np.diff(x.astype(np.int64)) < 0))
    return (
        f"count_descents n={n}",
        _time(lambda: np.count_nonzero(np.diff(x) < 0)),
        _time(lambda: ma.count_descents(x)),
    )


def main():
    print(f"{'case':<34}{'reference':>12}{'mojo-argsort':>16}{'ratio':>10}")
    print("-" * 72)
    for fn in (
        bench_int32,
        bench_int32_quicksort,
        bench_int64,
        bench_float64,
        bench_descending,
        bench_small,
        bench_counting,
        bench_lexsort,
        bench_descents,
    ):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<34}{ref*1e3:>10.2f}ms{got*1e3:>14.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
