"""Parity tests for mojo_argsort.

The upstream `argsort` distribution is not installed in this environment and
cannot be fetched, so the reference here is NumPy's own stable argsort, which
defines the same contract: permutation indices, stable among equal keys. Index
arrays are integers, so the comparison is exact -- `np.array_equal`, not
`assert_allclose`.
"""

import numpy as np
import pytest

import mojo_argsort as ma
from mojo_argsort._lib import INSERTION_LIMIT

ALL_DTYPES = [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64,
]


def _tie_heavy(rng, n, dtype):
    """Few distinct values, so tie-breaking is exercised, not just ordering."""
    if np.dtype(dtype).kind == "f":
        return rng.integers(-3, 4, size=n).astype(dtype)
    info = np.iinfo(dtype)
    lo = max(info.min, -3)
    hi = min(info.max, 3)
    return rng.integers(lo, hi + 1, size=n).astype(dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_matches_numpy_stable(dtype):
    rng = np.random.default_rng(1234)
    for n in (1, 2, 3, 17, 63, 64, 65, 129, 1000, 5000):
        x = _tie_heavy(rng, n, dtype)
        got = ma.argsort(x)
        expect = np.argsort(x, kind="stable")
        assert np.array_equal(got, expect), f"n={n} dtype={np.dtype(dtype).name}"


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_sorted_values_match(dtype):
    """Catches a wrong digit that still yields a permutation-ish sequence."""
    rng = np.random.default_rng(7)
    x = _tie_heavy(rng, 4099, dtype)
    as_float = np.asarray(x, dtype=np.float64)
    idx = ma.argsort(x)
    assert np.array_equal(np.sort(as_float), as_float[idx])


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_descending_is_negated_argsort(dtype):
    """Descending complements the mapped key, so ties keep input order."""
    rng = np.random.default_rng(99)
    x = _tie_heavy(rng, 3000, dtype)
    got = ma.argsort(x, descending=True)
    assert np.array_equal(got, np.argsort(-x, kind="stable"))


def test_descending_int64_matches_python_arithmetic():
    """`numpy` negation wraps on INT64_MIN, so the reference is Python ints."""
    x = np.array(
        [np.iinfo(np.int64).max, np.iinfo(np.int64).min, 0, -1, 1,
         np.iinfo(np.int64).min + 1, np.iinfo(np.int64).max - 1],
        dtype=np.int64,
    )
    expect = np.array(sorted(range(x.size), key=lambda i: -int(x[i])))
    assert np.array_equal(ma.argsort(x, descending=True), expect)


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_descending_unsigned(dtype):
    """Unsigned descending maps to the complemented key space."""
    rng = np.random.default_rng(101)
    x = rng.integers(0, 200, size=1000).astype(dtype)
    got = ma.argsort(x, descending=True)
    expect = np.argsort(np.asarray(x, dtype=np.int64).max() - x, kind="stable")
    assert np.array_equal(got, expect)
    values = x[got]
    assert np.all(values[:-1] >= values[1:])


def test_descending_ties_keep_input_order():
    x = np.array([1, 2, 1, 2, 1], dtype=np.int64)
    assert np.array_equal(
        ma.argsort(x, descending=True), np.argsort(-x, kind="stable")
    )


@pytest.mark.parametrize("n", [INSERTION_LIMIT, INSERTION_LIMIT + 1, 257])
def test_methods_agree(n):
    """The dispatch threshold must not change the answer, only the kernel."""
    rng = np.random.default_rng(5)
    x = rng.integers(-50, 50, size=n)
    assert np.array_equal(
        ma.argsort(x, method="insertion"), ma.argsort(x, method="radix")
    )
    assert np.array_equal(
        ma.argsort(x, method="insertion"), np.argsort(x, kind="stable")
    )


def test_all_equal_is_identity():
    """A radix pass that is not stable would scramble this."""
    x = np.full(1000, 7, dtype=np.int64)
    assert np.array_equal(ma.argsort(x), np.arange(1000))


def test_high_byte_dominates():
    """Only the top radix pass can order these, so a missing pass is fatal."""
    x = np.array([0, 1 << 60, 1 << 40, 1 << 55, (1 << 63) - 1, 5], dtype=np.int64)
    expect = np.argsort(x, kind="stable")
    assert np.array_equal(ma.argsort(x), expect)
    assert np.array_equal(ma.argsort(x, method="insertion"), expect)


def test_extreme_values():
    x = np.array(
        [np.iinfo(np.int64).max, np.iinfo(np.int64).min, 0, -1, 1,
         np.iinfo(np.int64).min + 1, np.iinfo(np.int64).max - 1],
        dtype=np.int64,
    )
    assert np.array_equal(ma.argsort(x), np.argsort(x, kind="stable"))


def test_infinity_and_finite_values():
    x = np.array([np.inf, 2.0, -np.inf, 1.0, -1.0, 3.0], dtype=np.float64)
    assert np.array_equal(ma.argsort(x), np.argsort(x, kind="stable"))
    assert np.array_equal(ma.argsort(x, descending=True),
                          np.argsort(-x, kind="stable"))


def test_signed_zero_is_ordered_by_bit_pattern():
    """Documented divergence: this sort sees -0.0 < +0.0, NumPy sees them equal.

    The kernel sorts the transformed bit pattern, where -0.0 maps to the
    largest key. NumPy compares with IEEE `<`, where the two are equal, so a
    stable sort keeps them in input order instead.
    """
    assert np.array_equal(ma.argsort(np.array([0.0, -0.0])), [1, 0])
    assert np.array_equal(ma.argsort(np.array([-0.0, 0.0])), [0, 1])
    assert np.array_equal(ma.argsort(np.array([0.0, -0.0]), descending=True),
                          [0, 1])


def test_nan_follows_the_documented_bit_order():
    """NaN lands by bit pattern, not where NumPy parks it.

    Documented rather than emulated: the order is defined on the transformed
    word, complementing every bit of a negative float and flipping the sign bit
    of a positive one. A positive NaN therefore sorts above every finite number
    like NumPy's does, while a negative NaN sorts first, where NumPy would move
    it to the end.
    """
    x = np.array([1.0, np.nan, -np.nan, 2.0, -np.inf], dtype=np.float64)
    words = x.view(np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    mapped = np.where((words & sign) == sign, ~words, words ^ sign)
    expect = np.array(sorted(range(x.size), key=lambda i: int(mapped[i])))
    assert np.array_equal(ma.argsort(x), expect)


def test_non_contiguous_input():
    """A strided view must give the same answer as its copy."""
    rng = np.random.default_rng(11)
    base = rng.integers(0, 9, size=6000)
    view = base[::3]
    assert np.array_equal(ma.argsort(view), np.argsort(view, kind="stable"))
    assert np.array_equal(ma.argsort(view), ma.argsort(np.ascontiguousarray(view)))


def test_empty_and_singleton():
    assert ma.argsort(np.array([], dtype=np.int64)).size == 0
    assert np.array_equal(ma.argsort(np.array([3], dtype=np.int64)), [0])


def test_rejects_unsupported_input():
    with pytest.raises(TypeError):
        ma.argsort(np.array([1, 2, 3], dtype=object))
    with pytest.raises(ValueError):
        ma.argsort(np.zeros((2, 3), dtype=np.float64))
    with pytest.raises(ValueError):
        ma.argsort(np.arange(10), method="quicksort")
