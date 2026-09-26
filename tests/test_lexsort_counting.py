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
