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


# --- Parity additions ---------------------------------------------------------
# The upstream `argsort` package is not installed here (verified: importing it
# fails with ModuleNotFoundError), so the reference is numpy's stable argsort,
# which README records as defining the same contract. Integer index arrays are
# compared exactly; the one floating-point value comparison below writes its
# tolerance explicitly (rtol=0, atol=0), i.e. exact, because no arithmetic on
# floats precedes it.


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_full_range_matches_numpy_stable(dtype):
    """Random full-range values, not just tie-heavy small values."""
    rng = np.random.default_rng(2026)
    kind = np.dtype(dtype).kind
    if kind == "f":
        x = rng.standard_normal(2000).astype(dtype)
    elif dtype == np.int64:
        x = rng.integers(0, 2**64, size=2000, dtype=np.uint64).view(np.int64)
    elif dtype == np.uint64:
        x = rng.integers(0, 2**64, size=2000, dtype=np.uint64)
    else:
        info = np.iinfo(dtype)
        x = rng.integers(info.min, info.max, size=2000).astype(dtype)
    assert np.array_equal(ma.argsort(x), np.argsort(x, kind="stable"))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_result_is_int64_permutation(dtype):
    rng = np.random.default_rng(11)
    x = rng.integers(-5, 5, size=100).astype(dtype) if np.dtype(dtype).kind != "f" else rng.standard_normal(100).astype(dtype)
    got = ma.argsort(x)
    assert got.dtype == np.int64
    assert sorted(int(i) for i in got) == list(range(x.size))


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
)
def test_descending_extremes_match_python_ints(dtype):
    """Descending at each width's extremes; numpy negation wraps at INT64_MIN."""
    info = np.iinfo(dtype)
    vals = [info.min, info.max, 0, 1]
    if info.min < 0:
        vals.append(-1)
        vals.append(info.min + 1)
    else:
        vals.append(2)
    x = np.array(vals, dtype=dtype)
    expect = np.array(sorted(range(x.size), key=lambda i: -int(x[i])))
    assert np.array_equal(ma.argsort(x, descending=True), expect)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_descending_infinity_matches_negated_numpy(dtype):
    x = np.array([np.inf, 2.5, -np.inf, -1.0, 0.5], dtype=dtype)
    assert np.array_equal(
        ma.argsort(x, descending=True), np.argsort(-x, kind="stable")
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_sorted_values_match_with_written_tolerance(dtype):
    """Sorted float values match numpy's up to an explicit zero tolerance."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000).astype(dtype)
    idx = ma.argsort(x)
    # Exactness is pinned here: rtol=0 and atol=0, because the permutation is
    # index arithmetic and no floating-point computation intervenes.
    np.testing.assert_allclose(
        x[idx].astype(np.float64),
        np.sort(x).astype(np.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert np.array_equal(idx, np.argsort(x, kind="stable"))


def test_duplicates_are_stable_for_both_methods():
    """Equal keys keep input order in ascending and descending order."""
    x = np.array([2, 1, 2, 1, 2, 1], dtype=np.int64)
    expect = np.argsort(x, kind="stable")
    for method in ("radix", "insertion", "auto"):
        assert np.array_equal(ma.argsort(x, method=method), expect)
        assert np.array_equal(ma.argsort(x, method=method, descending=True),
                              np.argsort(-x, kind="stable"))


def test_empty_all_dtypes_and_methods():
    for dtype in ALL_DTYPES:
        e = np.array([], dtype=dtype)
        assert ma.argsort(e).size == 0
        assert ma.argsort(e, method="radix").size == 0
        assert ma.argsort(e, method="insertion").size == 0
        assert ma.argsort(e).dtype == np.int64
    assert ma.argsort(np.array([9], dtype=np.int64), method="radix").tolist() == [0]
    assert ma.argsort(np.array([9], dtype=np.int64), method="insertion").tolist() == [0]


def test_nan_payloads_follow_bit_pattern_not_input_order():
    """Distinct positive NaN payloads order by payload, unlike numpy stable.

    This is the same documented bit-pattern divergence as negative NaN: the
    mapped words of the two payloads differ, so no tie remains for stability
    to break.
    """
    w1 = np.uint64(0x7FF8000000000001)
    w2 = np.uint64(0x7FF8000000000002)
    x = np.array([w2, w1], dtype=np.uint64).view(np.float64)
    assert np.array_equal(ma.argsort(x), [1, 0])
    # The divergence is real, not a silent match: numpy would keep input order.
    assert np.array_equal(np.argsort(x, kind="stable"), [0, 1])
    assert np.array_equal(ma.argsort(x, method="insertion"), [1, 0])
    y = np.array([w1, w2], dtype=np.uint64).view(np.float64)
    assert np.array_equal(ma.argsort(y), [0, 1])


def test_reversed_stride_input():
    rev = np.arange(10, dtype=np.float64)[::-1]
    assert np.array_equal(ma.argsort(rev), np.argsort(rev, kind="stable"))
    assert np.array_equal(ma.argsort(rev), np.arange(9, -1, -1))


def test_degenerate_rejections_match_upstream_scope():
    """Non-numeric and non-scalar shapes raise the port's documented classes."""
    # numpy argsort sorts bool/complex/strings, but the port's documented
    # scope is numeric 1-D only, so rejection (not a value) is the contract.
    for bad in (
        np.array([True, False]),
        np.array([1 + 2j, 3 + 0j]),
        np.array(["b", "a"]),
    ):
        with pytest.raises(TypeError):
            ma.argsort(bad)
    with pytest.raises(ValueError):
        ma.argsort(np.array(5))
    with pytest.raises(ValueError):
        ma.argsort(np.arange(3), method=None)


def test_input_is_not_mutated():
    x = np.array([3.0, 1.0, 2.0], dtype=np.float64)
    keep = x.copy()
    ma.argsort(x)
    ma.argsort(x, descending=True)
    ma.argsort(x, method="insertion")
    assert np.array_equal(x, keep)


# --- TEST-phase parity audit ---------------------------------------------------
# Upstream `argsort` is not installed (verified: import argsort raises
# ModuleNotFoundError under pixi run python), so the reference is
# numpy.argsort(kind="stable") for argsort, np.lexsort for lexsort, Python
# integer arithmetic for counting_argsort/count_descents/is_sorted.
# Every comparison below composes the port with the upstream reference instead
# of comparing the port to a hand-written literal. Index arrays are exact
# (np.array_equal); every float-value or integer-count comparison writes its
# tolerance explicitly as rtol=0.0, atol=0.0 (exact), because the kernel does
# no floating-point arithmetic.


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_argsort_composes_with_numpy_reference(dtype):
    """argsort(x) == np.argsort(x, kind="stable") index-for-index."""
    rng = np.random.default_rng(2027)
    if np.dtype(dtype).kind == "f":
        x = rng.standard_normal(500).astype(dtype)
    else:
        info = np.iinfo(dtype)
        x = rng.integers(max(info.min, -1000), min(info.max, 1000) + 1, size=500).astype(dtype)
    got, expect = ma.argsort(x), np.argsort(x, kind="stable")
    assert np.array_equal(got, expect)
    assert np.array_equal(x[got], x[expect])
    if np.dtype(dtype).kind == "f":
        np.testing.assert_allclose(
            np.sort(x.astype(np.float64)),
            x[got].astype(np.float64),
            rtol=0.0, atol=0.0,
        )


def test_argsort_primary_sorted_values_with_written_tolerance():
    rng = np.random.default_rng(2028)
    for dtype in (np.float16, np.float32, np.float64):
        x = rng.standard_normal(1000).astype(dtype)
        idx = ma.argsort(x)
        np.testing.assert_allclose(
            x[idx].astype(np.float64), np.sort(x).astype(np.float64),
            rtol=0.0, atol=0.0,
        )


def test_argsort_degenerate_compositions():
    """Empty, singleton, duplicates: same values as the numpy reference."""
    for dtype in ALL_DTYPES:
        e = np.array([], dtype=dtype)
        assert np.array_equal(ma.argsort(e), np.argsort(e, kind="stable"))
        s = np.array([7], dtype=dtype)
        assert np.array_equal(ma.argsort(s), np.argsort(s, kind="stable"))
        assert np.array_equal(ma.argsort(s, descending=True), [0])
    dup = np.array([2, 1, 2, 1, 2, 1], dtype=np.int64)
    assert np.array_equal(ma.argsort(dup), np.argsort(dup, kind="stable"))
    assert np.array_equal(ma.argsort(dup, descending=True),
                          np.argsort(-dup, kind="stable"))


def test_argsort_nan_inf_all_dtypes_compositions():
    """Finite/inf order matches numpy; NaN order follows mapped bits."""
    x = np.array([np.inf, 2.0, -np.inf, 1.0, -1.0], dtype=np.float64)
    assert np.array_equal(ma.argsort(x), np.argsort(x, kind="stable"))
    assert np.array_equal(ma.argsort(x, descending=True),
                          np.argsort(-x, kind="stable"))
    # NaN: positive NaN above finite like numpy, negative NaN first (documented).
    y = np.array([1.0, np.nan, -np.nan, 2.0], dtype=np.float64)
    words = y.view(np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    mapped = np.where((words & sign) == sign, ~words, words ^ sign)
    expect = np.array(sorted(range(y.size), key=lambda i: int(mapped[i])))
    assert np.array_equal(ma.argsort(y), expect)


def test_argsort_mismatched_shapes_raise_same_class_as_upstream_scope():
    """2-D/0-D raise ValueError (port scope), non-numeric raise TypeError."""
    with pytest.raises(ValueError):
        ma.argsort(np.zeros((2, 3), dtype=np.float64))
    with pytest.raises(ValueError):
        ma.argsort(np.array(5))
    with pytest.raises(TypeError):
        ma.argsort(np.array([True, False]))
    with pytest.raises(TypeError):
        ma.argsort(np.array([1 + 2j, 3 + 0j]))
    with pytest.raises(TypeError):
        ma.argsort(np.array(["b", "a"]))
