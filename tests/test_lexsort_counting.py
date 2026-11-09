"""Tests for the multi-key and counting-sort surfaces of mojo_argsort."""

import numpy as np
import pytest

import mojo_argsort as ma


def test_lexsort_two_keys_matches_numpy():
    rng = np.random.default_rng(3)
    for n in (1, 5, 64, 65, 2000):
        a = rng.integers(0, 4, size=n)
        b = rng.integers(0, 3, size=n)
        got = ma.lexsort(b, a)
        expect = np.lexsort((b, a))
        assert np.array_equal(got, expect), f"n={n}"


def test_lexsort_three_keys_matches_numpy():
    """Only the last key is dominant, so a wrong iteration order shows here."""
    rng = np.random.default_rng(4)
    a = rng.integers(0, 3, size=997)
    b = rng.integers(0, 3, size=997)
    c = rng.integers(0, 3, size=997)
    got = ma.lexsort(a, b, c)
    expect = np.lexsort((a, b, c))
    assert np.array_equal(got, expect)


def test_lexsort_mixed_dtypes():
    rng = np.random.default_rng(6)
    a = rng.integers(-5, 5, size=500)
    b = rng.standard_normal(500)
    c = rng.integers(0, 2, size=500)
    assert np.array_equal(ma.lexsort(a, b, c), np.lexsort((a, b, c)))


def test_lexsort_descending_flips_every_key():
    rng = np.random.default_rng(8)
    a = rng.integers(0, 3, size=300)
    b = rng.integers(0, 3, size=300)
    got = ma.lexsort(b, a, descending=True)
    expect = np.lexsort((-b, -a))
    assert np.array_equal(got, expect)


def test_lexsort_rejects_bad_input():
    with pytest.raises(ValueError):
        ma.lexsort()
    with pytest.raises(ValueError):
        ma.lexsort(np.arange(3), np.arange(4))


@pytest.mark.parametrize("buckets", [1, 2, 8, 256, 1024])
def test_counting_argsort_matches_python_stable(buckets):
    rng = np.random.default_rng(buckets)
    x = rng.integers(0, 5000, size=2000).astype(np.int32)
    got = ma.counting_argsort(x, buckets)
    expect = sorted(range(x.size), key=lambda i: int(x[i]) % buckets)
    assert np.array_equal(got, np.array(expect, dtype=np.int64))


def test_counting_argsort_descending():
    x = np.array([9, 3, 12, 5, 7, 1, 20, 2], dtype=np.int32)
    got = ma.counting_argsort(x, 4, descending=True)
    # descending orders the remainder from high to low, ties keep input order
    expect = sorted(range(x.size), key=lambda i: -(int(x[i]) % 4))
    assert np.array_equal(got, np.array(expect, dtype=np.int64))


def test_counting_argsort_single_bucket_is_identity():
    x = np.array([5, 1, 9, 3], dtype=np.int32)
    assert np.array_equal(ma.counting_argsort(x, 1), np.arange(4))


def test_counting_argsort_rejects_16bit():
    with pytest.raises(TypeError):
        ma.counting_argsort(np.arange(10, dtype=np.int16), 4)


def test_count_descents_matches_numpy():
    rng = np.random.default_rng(13)
    for dtype in (np.int32, np.int64, np.float64, np.uint16, np.float32):
        x = np.ascontiguousarray(rng.integers(-3, 4, size=777).astype(dtype))
        # widen before differencing: numpy's diff wraps on unsigned input
        as_int = x.astype(np.int64)
        expect = int(np.count_nonzero(np.diff(as_int) < 0))
        assert ma.count_descents(x) == expect, np.dtype(dtype).name
        up = int(np.count_nonzero(np.diff(as_int) > 0))
        assert ma.count_descents(x, descending=True) == up


def test_count_descents_edge_cases():
    assert ma.count_descents(np.array([], dtype=np.int64)) == 0
    assert ma.count_descents(np.array([7], dtype=np.int64)) == 0
    assert ma.count_descents(np.array([1, 2, 3], dtype=np.int64)) == 0
    assert ma.count_descents(np.array([1, 1, 2], dtype=np.int64)) == 0
    assert ma.count_descents(np.array([3, 1], dtype=np.int64)) == 1
    assert ma.count_descents(np.array([1, 1, 1], dtype=np.int64)) == 0


def test_is_sorted_agrees_with_numpy():
    rng = np.random.default_rng(17)
    x = np.sort(rng.integers(0, 50, size=400))
    assert ma.is_sorted(x)
    assert ma.count_descents(x) == 0
    y = x.copy()
    y[10], y[200] = y[200], y[10]
    assert not ma.is_sorted(y)
    assert not ma.is_sorted(x, descending=True)


# --- Parity additions ---------------------------------------------------------
# Upstream `argsort` is not installed; numpy stable argsort/lexsort plus
# Python integer arithmetic are the reference, as above. Every float-value
# comparison below writes its tolerance (rtol=0, atol=0, exact) since no
# floating-point arithmetic intervenes.


def _mapped_float64(x):
    words = np.asarray(x, dtype=np.float64).view(np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    return np.where((words & sign) == sign, ~words, words ^ sign)


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
)
def test_count_descents_all_int_dtypes_match_numpy_diff(dtype):
    rng = np.random.default_rng(29)
    if np.dtype(dtype).kind == "i":
        x = rng.integers(-1000, 1000, size=777).astype(dtype)
    else:
        x = rng.integers(0, 1000, size=777).astype(dtype)
    as_int = x.astype(np.int64)
    assert ma.count_descents(x) == int(np.count_nonzero(np.diff(as_int) < 0))
    assert ma.count_descents(x, descending=True) == int(np.count_nonzero(np.diff(as_int) > 0))
    assert ma.is_sorted(x) == (int(np.count_nonzero(np.diff(as_int) < 0)) == 0)


def test_count_descents_finite_floats_match_numpy_diff():
    """Finite floats without -0.0/NaN: mapped order and diff agree."""
    rng = np.random.default_rng(31)
    for dtype in (np.float16, np.float32, np.float64):
        x = rng.standard_normal(777).astype(dtype)
        as_float = x.astype(np.float64)
        assert ma.count_descents(x) == int(np.count_nonzero(np.diff(as_float) < 0))
        assert ma.count_descents(x, descending=True) == int(np.count_nonzero(np.diff(as_float) > 0))
        idx = ma.argsort(x)
        np.testing.assert_allclose(
            as_float[idx], np.sort(as_float), rtol=0.0, atol=0.0
        )


def test_lexsort_float_keys_with_infinity_match_numpy():
    a = np.array([np.inf, 1.0, -np.inf, 1.0, np.inf], dtype=np.float64)
    b = np.array([2, 0, 0, 1, 1], dtype=np.int64)
    assert np.array_equal(ma.lexsort(b, a), np.lexsort((b, a)))
    c = np.array([0.5, -np.inf, -np.inf, 0.5, -1.0], dtype=np.float32)
    assert np.array_equal(ma.lexsort(c, a, b), np.lexsort((c, a, b)))


def test_lexsort_descending_int64_min_matches_python_ints():
    """numpy negation wraps on INT64_MIN, so the reference is Python ints."""
    x = np.array([np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max], dtype=np.int64)
    expect = np.array(sorted(range(x.size), key=lambda i: -int(x[i])))
    assert np.array_equal(ma.lexsort(x, descending=True), expect)
    a = np.array([np.iinfo(np.int64).min, 1, 0], dtype=np.int64)
    b = np.array([0, np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype=np.int64)
    expect2 = np.array(sorted(range(3), key=lambda i: (-int(a[i]), -int(b[i]))))
    assert np.array_equal(ma.lexsort(b, a, descending=True), expect2)


def test_lexsort_signed_zero_divergence_is_explicit():
    """-0.0 sorts below +0.0 by bit pattern, so a tie-break cannot fire.

    numpy compares -0.0 == +0.0 and breaks the tie on the secondary key;
    the port orders on the primary key's bits instead. Pinned so the
    divergence is a test, not an accident.
    """
    a = np.array([0.0, -0.0], dtype=np.float64)
    b = np.array([0, 1], dtype=np.int64)
    assert np.array_equal(ma.lexsort(b, a), [1, 0])
    assert np.array_equal(np.lexsort((b, a)), [0, 1])
    c = np.array([-0.0, 0.0], dtype=np.float64)
    assert np.array_equal(ma.lexsort(b, c), [0, 1])
    assert np.array_equal(np.lexsort((b, c)), [0, 1])


def test_lexsort_empty_and_singleton():
    e = np.array([], dtype=np.int64)
    assert ma.lexsort(e).size == 0
    assert ma.lexsort(e, e).size == 0
    assert np.array_equal(ma.lexsort(e), np.lexsort((e,)))
    one = np.array([5], dtype=np.int64)
    one_b = np.array([3], dtype=np.int64)
    assert np.array_equal(ma.lexsort(one_b, one), [0])
    assert np.array_equal(ma.lexsort(one), np.argsort(one, kind="stable"))


def test_lexsort_rejects_non_1d_and_non_numeric():
    with pytest.raises(ValueError):
        ma.lexsort(np.zeros((2, 3)), np.arange(6))
    with pytest.raises(ValueError):
        ma.lexsort(np.array(5))
    with pytest.raises(TypeError):
        ma.lexsort(np.array([True, False]), np.array([1, 0]))
    with pytest.raises(TypeError):
        ma.lexsort(np.array(["b", "a"]), np.array([1, 0]))


def test_counting_argsort_int64_and_uint64_match_python_modulo():
    """64-bit keys bucket by value % buckets, as 32-bit keys do.

    Before the fix the kernel read each 64-bit key as two 32-bit words, so
    only the first n/2 values influenced the order. These cases separate
    that bug from correct modulo bucketing.
    """
    rng = np.random.default_rng(37)
    for dtype in (np.int64, np.uint64):
        for buckets in (2, 4, 8, 256):
            x = rng.integers(0, 10000, size=2000).astype(dtype)
            expect = np.array(sorted(range(x.size), key=lambda i: int(x[i]) % buckets))
            assert np.array_equal(ma.counting_argsort(x, buckets), expect)


def test_counting_argsort_extremes_match_python_modulo():
    x = np.array(
        [np.iinfo(np.int32).min, np.iinfo(np.int32).max, -1, 0, 1], dtype=np.int32
    )
    for buckets in (2, 4, 8, 256):
        expect = np.array(sorted(range(x.size), key=lambda i: int(x[i]) % buckets))
        assert np.array_equal(ma.counting_argsort(x, buckets), expect)
    y = np.array(
        [np.iinfo(np.uint32).max, 0, 1, np.iinfo(np.uint32).min], dtype=np.uint32
    )
    for buckets in (4, 16):
        expect = np.array(sorted(range(y.size), key=lambda i: int(y[i]) % buckets))
        assert np.array_equal(ma.counting_argsort(y, buckets), expect)
    z = np.array([2**32 + 1, 2**32 + 2, 1, 2], dtype=np.uint64)
    expect = np.array(sorted(range(z.size), key=lambda i: int(z[i]) % 4))
    assert np.array_equal(ma.counting_argsort(z, 4), expect)


def test_counting_argsort_negative_descending():
    for dtype in (np.int32, np.int64):
        x = np.array([-1, -2, -3, -4, 0, 1], dtype=dtype)
        for buckets in (4, 8):
            expect = np.array(
                sorted(range(x.size), key=lambda i: -(int(x[i]) % buckets))
            )
            assert np.array_equal(
                ma.counting_argsort(x, buckets, descending=True), expect
            )


def test_counting_argsort_float_keys_follow_mapped_bits():
    """Float buckets are mapped-bit remainders, pinned with rtol/atol=0."""
    x = np.array([1.0, 2.0, 4.0, 0.5, -1.0, np.inf], dtype=np.float64)
    mapped = _mapped_float64(x)
    for buckets in (4, 8):
        expect = np.array(
            sorted(range(x.size), key=lambda i: int(mapped[i]) % buckets)
        )
        got = ma.counting_argsort(x, buckets)
        assert np.array_equal(got, expect)
        # Bucket order check with an explicit tolerance: got's remainders
        # are non-decreasing. Cast uint64 remainders to float64 only for the
        # allclose call; values fit exactly below 2**53 for small test data,
        # and the tolerance is still rtol=0, atol=0.
        remainders = (mapped[got] % np.uint64(buckets)).astype(np.float64)
        assert np.all(remainders[1:] >= remainders[:-1])
        np.testing.assert_allclose(
            remainders,
            np.array(sorted(int(mapped[i]) % buckets for i in range(x.size)), dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
    x32 = np.array([1.5, -2.5, 0.5, 3.25], dtype=np.float32)
    words = x32.view(np.uint32)
    sign = np.uint32(0x80000000)
    mapped32 = np.where((words & sign) == sign, ~words, words ^ sign)
    expect = np.array(sorted(range(x32.size), key=lambda i: int(mapped32[i]) % 4))
    assert np.array_equal(ma.counting_argsort(x32, 4), expect)


def test_counting_argsort_empty_singleton_and_noncontiguous():
    e = np.array([], dtype=np.int32)
    assert ma.counting_argsort(e, 4).size == 0
    e64 = np.array([], dtype=np.int64)
    assert ma.counting_argsort(e64, 4).size == 0
    assert np.array_equal(ma.counting_argsort(np.array([7], dtype=np.int32), 4), [0])
    view = np.arange(20, dtype=np.int32)[::2]
    expect = np.array(sorted(range(view.size), key=lambda i: int(view[i]) % 4))
    assert np.array_equal(ma.counting_argsort(view, 4), expect)
    # Invalid buckets are rejected even for an empty input.
    with pytest.raises(ValueError):
        ma.counting_argsort(e, 3)
    with pytest.raises(ValueError):
        ma.counting_argsort(np.zeros((2, 3), dtype=np.int32), 4)
    with pytest.raises(TypeError):
        ma.counting_argsort(np.array([True, False]), 2)


def test_counting_argsort_nan_positions_follow_mapped_bits():
    """NaN keys bucket by mapped bits; finite neighbours still order."""
    x = np.array([1.0, np.nan, 2.0, -np.nan, -np.inf], dtype=np.float64)
    mapped = _mapped_float64(x)
    for buckets in (4, 8):
        expect = np.array(
            sorted(range(x.size), key=lambda i: int(mapped[i]) % buckets)
        )
        assert np.array_equal(ma.counting_argsort(x, buckets), expect)


def test_count_descents_nan_inf_and_signed_zero():
    """Mapped-key descents for specials, vs the mapped reference itself."""
    cases = [
        [1.0, np.nan],
        [np.nan, 1.0],
        [np.inf, 1.0],
        [1.0, np.inf],
        [np.inf, np.inf],
        [np.inf, np.nan],
        [np.nan, np.inf],
        [0.0, -0.0],
        [-0.0, 0.0],
        [1.0, np.nan, 0.0],
    ]
    for vals in cases:
        x = np.array(vals, dtype=np.float64)
        mapped = _mapped_float64(x)
        expect = int(np.sum(mapped[1:] < mapped[:-1]))
        assert ma.count_descents(x) == expect, vals
        assert ma.is_sorted(x) == (expect == 0), vals
    # numpy's diff reference would give 0 for [1.0, nan, 0.0]; the port's
    # mapped scan gives 1, which is the documented bit-pattern semantics.
    x = np.array([1.0, np.nan, 0.0], dtype=np.float64)
    assert int(np.count_nonzero(np.diff(x) < 0)) == 0
    assert ma.count_descents(x) == 1


def test_count_descents_nan_payloads():
    w1 = np.uint64(0x7FF8000000000001)
    w2 = np.uint64(0x7FF8000000000002)
    x = np.array([w1, w2], dtype=np.uint64).view(np.float64)
    y = np.array([w2, w1], dtype=np.uint64).view(np.float64)
    assert ma.count_descents(x) == 0
    assert ma.count_descents(y) == 1
    assert ma.is_sorted(x)
    assert not ma.is_sorted(y)


def test_count_descents_and_is_sorted_empty_singleton_descending():
    for dtype in (np.int64, np.float64, np.int32):
        e = np.array([], dtype=dtype)
        assert ma.count_descents(e) == 0
        assert ma.count_descents(e, descending=True) == 0
        assert ma.is_sorted(e)
        assert ma.is_sorted(e, descending=True)
        s = np.array([7], dtype=dtype)
        assert ma.count_descents(s) == 0
        assert ma.is_sorted(s)
        assert ma.is_sorted(s, descending=True)
    x = np.array([3, 2, 2, 1], dtype=np.int64)
    assert ma.is_sorted(x, descending=True)
    assert ma.count_descents(x, descending=True) == 0
    # Two strict falls (3>2 and 2>1) with one equal step between them.
    assert ma.count_descents(x) == int(np.count_nonzero(np.diff(x) < 0))
    assert ma.count_descents(x) == 2
    with pytest.raises(ValueError):
        ma.count_descents(np.zeros((2, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        ma.is_sorted(np.zeros((2, 3), dtype=np.int64))
    with pytest.raises(TypeError):
        ma.count_descents(np.array([True, False]))


# --- TEST-phase parity audit ---------------------------------------------------
# Upstream `argsort` is not installed (verified: import argsort raises
# ModuleNotFoundError under pixi run python), so the reference for lexsort is
# numpy.lexsort, and for counting_argsort/count_descents/is_sorted it is Python
# integer arithmetic. Each test below composes the port with that upstream
# reference rather than a hand-written literal. Index comparisons are exact
# (np.array_equal); every count/float-value comparison writes its tolerance
# explicitly as rtol=0.0, atol=0.0 (exact), because the kernel performs no
# floating-point arithmetic on these paths.


def test_lexsort_composes_with_numpy_reference():
    """lexsort(b, a) composes with np.lexsort((b, a)) and with argsort."""
    rng = np.random.default_rng(2029)
    for n in (0, 1, 2, 17, 64, 65, 1000):
        a = rng.integers(0, 4, size=n)
        b = rng.integers(0, 3, size=n)
        assert np.array_equal(ma.lexsort(b, a), np.lexsort((b, a))), f"n={n}"
    # Lexsort with one key composes with argsort (numpy's own equivalence).
    x = rng.integers(-9, 9, size=500)
    assert np.array_equal(ma.lexsort(x), ma.argsort(x))
    assert np.array_equal(ma.lexsort(x), np.argsort(x, kind="stable"))


def test_lexsort_float_keys_sorted_values_with_written_tolerance():
    rng = np.random.default_rng(2030)
    a = rng.standard_normal(1000)
    b = rng.integers(0, 3, size=1000)
    idx = ma.lexsort(b, a)
    assert np.array_equal(idx, np.lexsort((b, a)))
    np.testing.assert_allclose(a[idx], np.sort(a), rtol=0.0, atol=0.0)
    assert np.array_equal(a[idx], np.sort(a))


def test_lexsort_nan_inf_with_mapped_bit_reference():
    """NaN/inf lexsort keys follow mapped bits, matching argsort."""
    x = np.array([1.0, np.nan, -np.nan, 2.0, -np.inf], dtype=np.float64)
    b = np.zeros(5, dtype=np.int64)
    words = x.view(np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    mapped = np.where((words & sign) == sign, ~words, words ^ sign)
    expect = np.array(sorted(range(x.size), key=lambda i: int(mapped[i])))
    assert np.array_equal(ma.lexsort(b, x), expect)
    assert np.array_equal(ma.lexsort(b, x), ma.argsort(x))
    # Finite/inf keys without NaN or signed zero compose with numpy.
    y = np.array([np.inf, 1.0, -np.inf, 1.0, 2.5], dtype=np.float64)
    c = np.array([2, 0, 0, 1, 1], dtype=np.int64)
    assert np.array_equal(ma.lexsort(c, y), np.lexsort((c, y)))


def test_lexsort_degenerate_compositions():
    """Empty/singleton/duplicates compose with numpy; mismatches raise."""
    assert np.array_equal(ma.lexsort(np.array([], dtype=np.int64)),
                          np.lexsort((np.array([], dtype=np.int64),)))
    one_b, one_a = np.array([3], dtype=np.int64), np.array([5], dtype=np.int64)
    assert np.array_equal(ma.lexsort(one_b, one_a), [0])
    dup_a = np.array([1, 1, 1], dtype=np.int64)
    dup_b = np.array([2, 2, 2], dtype=np.int64)
    assert np.array_equal(ma.lexsort(dup_b, dup_a), np.lexsort((dup_b, dup_a)))
    assert np.array_equal(ma.lexsort(dup_b, dup_a), np.arange(3))
    # Mismatched lengths: same error class as np.lexsort (ValueError).
    with pytest.raises(ValueError):
        ma.lexsort(np.arange(3), np.arange(4))
    with pytest.raises(ValueError):
        np.lexsort((np.arange(3), np.arange(4)))
    with pytest.raises(ValueError):
        ma.lexsort()
    with pytest.raises(ValueError):
        ma.lexsort(np.zeros((2, 3)), np.arange(6))
    with pytest.raises(TypeError):
        ma.lexsort(np.array([True, False]), np.array([1, 0]))


def test_counting_argsort_composes_with_python_modulo_reference():
    """counting_argsort(x, b) == indices sorted by x % b (Python ints)."""
    rng = np.random.default_rng(2031)
    for dtype in (np.int32, np.uint32, np.int64, np.uint64):
        for buckets in (1, 2, 4, 8, 256, 1024):
            x = rng.integers(0, 10000, size=1000).astype(dtype)
            expect = np.array(sorted(range(x.size), key=lambda i: int(x[i]) % buckets))
            got = ma.counting_argsort(x, buckets)
            assert np.array_equal(got, expect)
            remainders = (x[got] % buckets).astype(np.float64)
            np.testing.assert_allclose(
                remainders,
                np.array(sorted(int(v) % buckets for v in x), dtype=np.float64),
                rtol=0.0, atol=0.0,
            )


def test_counting_argsort_degenerate_compositions():
    """Empty, singleton, duplicates, NaN/inf and bad buckets."""
    e = np.array([], dtype=np.int32)
    assert ma.counting_argsort(e, 4).size == 0
    assert np.array_equal(ma.counting_argsort(np.array([7], dtype=np.int32), 4), [0])
    dup = np.array([7, 7, 7, 7], dtype=np.int32)
    assert np.array_equal(ma.counting_argsort(dup, 4), np.arange(4))
    assert np.array_equal(ma.counting_argsort(dup, 4, descending=True), np.arange(4))
    x = np.array([1.0, np.nan, 2.0], dtype=np.float64)
    mapped = _mapped_float64(x)
    expect = np.array(sorted(range(x.size), key=lambda i: int(mapped[i]) % 4))
    assert np.array_equal(ma.counting_argsort(x, 4), expect)
    inf = np.array([np.inf, 1.0, -np.inf], dtype=np.float64)
    mapped_inf = _mapped_float64(inf)
    expect_inf = np.array(sorted(range(inf.size), key=lambda i: int(mapped_inf[i]) % 8))
    assert np.array_equal(ma.counting_argsort(inf, 8), expect_inf)
    for bad in (0, 3, -4, 6):
        with pytest.raises(ValueError):
            ma.counting_argsort(np.array([1, 2], dtype=np.int32), bad)
    with pytest.raises(ValueError):
        ma.counting_argsort(np.zeros((2, 3), dtype=np.int32), 4)
    with pytest.raises(TypeError):
        ma.counting_argsort(np.array([True, False]), 2)


def test_count_descents_composes_with_numpy_diff_reference():
    """count_descents == count_nonzero(diff < 0), tolerance rtol/atol=0."""
    rng = np.random.default_rng(2032)
    # Why the reference is widened: np.diff on uint64 wraps (e.g.
    # np.diff(np.array([2**64-1, 5], dtype=np.uint64)) is 6, not a negative),
    # and np.diff on int8 wraps too, so the reference widens to int64/float64
    # before differencing -- exactly as the pre-existing
    # test_count_descents_matches_numpy does ("widen before differencing").
    # The port counts descents on mapped keys directly, so its result equals
    # the widened count, not the naive in-dtype diff count.
    for dtype in (np.int8, np.int16, np.int32, np.int64,
                  np.uint8, np.uint16, np.uint32, np.uint64,
                  np.float16, np.float32, np.float64):
        if np.dtype(dtype).kind == "f":
            x = rng.standard_normal(777).astype(dtype)
        elif np.dtype(dtype).kind == "u":
            x = rng.integers(0, 100, size=777).astype(dtype)
        else:
            x = rng.integers(-100, 100, size=777).astype(dtype)
        if np.dtype(dtype).kind == "f":
            expect = int(np.count_nonzero(np.diff(x.astype(np.float64)) < 0))
        elif np.dtype(dtype).kind == "u":
            # uint64 values >= 2**63 cannot be widened to int64; compare
            # adjacent elements directly (no subtraction, so no wrap).
            expect = int(np.count_nonzero(x[1:] < x[:-1]))
        else:
            expect = int(np.count_nonzero(np.diff(x.astype(np.int64)) < 0))
        np.testing.assert_allclose(float(ma.count_descents(x)), float(expect),
                                   rtol=0.0, atol=0.0)
        assert ma.count_descents(x) == expect


def test_count_descents_degenerate_compositions():
    """Empty/singleton/duplicates/NaN-inf compose with mapped reference."""
    assert ma.count_descents(np.array([], dtype=np.int64)) == 0
    assert ma.count_descents(np.array([7], dtype=np.int64)) == 0
    dup = np.array([2, 2, 1, 1, 3], dtype=np.int64)
    expect = int(np.count_nonzero(np.diff(dup) < 0))
    assert ma.count_descents(dup) == expect == 1
    x = np.array([1.0, np.nan, 0.0], dtype=np.float64)
    mapped = _mapped_float64(x)
    assert ma.count_descents(x) == int(np.sum(mapped[1:] < mapped[:-1])) == 1
    inf = np.array([np.inf, 1.0, -np.inf, 0.0], dtype=np.float64)
    mapped_inf = _mapped_float64(inf)
    assert ma.count_descents(inf) == int(np.sum(mapped_inf[1:] < mapped_inf[:-1]))
    with pytest.raises(ValueError):
        ma.count_descents(np.zeros((2, 3), dtype=np.int64))
    with pytest.raises(TypeError):
        ma.count_descents(np.array([True, False]))


def test_is_sorted_composes_with_count_descents_reference():
    """is_sorted == (count_descents == 0), tolerance rtol/atol=0."""
    rng = np.random.default_rng(2033)
    for dtype in (np.int32, np.int64, np.float32, np.float64):
        x = rng.integers(-9, 9, size=300).astype(dtype) \
            if np.dtype(dtype).kind != "f" else rng.standard_normal(300).astype(dtype)
        expect_desc = int(np.count_nonzero(
            np.diff(x.astype(np.int64)) < 0)) if np.dtype(dtype).kind != "f" \
            else int(np.count_nonzero(np.diff(x.astype(np.float64)) < 0))
        np.testing.assert_allclose(float(ma.is_sorted(x)),
                                   float(expect_desc == 0), rtol=0.0, atol=0.0)
        assert ma.is_sorted(x) == (ma.count_descents(x) == 0)
        assert ma.is_sorted(x, descending=True) == (ma.count_descents(x, descending=True) == 0)
    assert ma.is_sorted(np.array([], dtype=np.int64))
    assert ma.is_sorted(np.array([7], dtype=np.int64))
    assert ma.is_sorted(np.array([1, 1, 2], dtype=np.int64))
    assert ma.is_sorted(np.array([2, 1, 1], dtype=np.int64), descending=True)
    with pytest.raises(ValueError):
        ma.is_sorted(np.zeros((2, 3), dtype=np.int64))
    with pytest.raises(TypeError):
        ma.is_sorted(np.array([True, False]))
