# mojo-argsort

`mojo-argsort` is the compute-oriented subset of an argsort library with the
sorting inner loops written in Mojo and callable from Python. The Python package
is named `mojo_argsort`, so it installs alongside any real `argsort`
distribution rather than shadowing it.

```python
import numpy as np
import mojo_argsort as ma

x = np.array([3.0, -1.0, 2.0, -1.0])
ma.argsort(x)                       # array([1, 3, 2, 0])
ma.argsort(x, descending=True)      # array([0, 2, 1, 3])
ma.lexsort(y, x)                    # lexicographic, x dominant
ma.count_descents(x)                # 2
```

## What is ported, and why

Sorting *is* the numeric core of an argsort library: the deliverable is a
permutation, which means a loop over the keys, a comparison or a digit
extraction per element, and repeated passes over the data. Everything here is
that loop, moved into compiled code.

| area | implemented API | kernel |
| --- | --- | --- |
| Single-key index sort | `argsort(x)` | `mas_radix_argsort` (LSD radix, 8 passes of 8 bits at 64-bit, 4 at 32-bit) |
| Single-key index sort, short inputs | `argsort(x, method="insertion")` | `mas_insertion_argsort` (stable insertion sort) |
| Descending order | `argsort(x, descending=True)` | same kernels, complemented key |
| Multi-key index sort | `lexsort(*keys)` | chain of stable `mas_radix_argsort` calls |
| Bucketed index sort | `counting_argsort(x, buckets)` | `mas_counting_argsort` (histogram, prefix sum, scatter) |
| Sortedness statistic | `count_descents(x)`, `is_sorted(x)` | `mas_count_descents` (single scan) |

`count_descents` has no direct NumPy equivalent. It is the natural-run count a
merge or MSD radix sort consults before choosing a strategy, and it is a real
loop over the keys, so it belongs here.

## Not implemented

- Comparisons on Python objects, strings or records. The kernels work on the raw
  bytes of a numeric array; a key function is dispatch overhead, not compute.
- Multi-dimensional `axis` handling. Everything is 1-D; reshape before calling.
- Comparison sorts for large non-numeric inputs (quicksort, heapsort, Timsort).
  Only the radix and insertion kernels exist.
- Sorting a list of tuples or a `list` of arbitrary Python values.
- NaN-ordering parity with NumPy: see below.

## Parity and its limits

The parity reference for the `argsort` distribution could not be used: it is not
installed in this environment and the box has no network to fetch it. Parity is
therefore asserted against `numpy`, whose `argsort(kind="stable")` defines the
same contract (permutation indices, stable among equal keys), and against Python
integer arithmetic where NumPy's own reference is wrong.

Parity is **exact**, not approximate: index arrays are integers, so the tests use
`np.array_equal` with no tolerance. There is no floating-point arithmetic in the
kernels at all, so there is nothing for FMA to perturb.

Two documented divergences from NumPy, both consequences of sorting the bit
pattern rather than the value:

- **Signed zero.** `-0.0` maps below `+0.0`. NumPy compares with IEEE `<`, where
  they are equal, and a stable sort therefore keeps them in input order.
- **NaN.** A positive NaN sorts above every finite number, as in NumPy. A
  negative NaN sorts first, where NumPy moves every NaN to the end.

The key map is the reason one kernel serves six dtypes. Two's complement and
IEEE-754 do *not* share a transform, so there are three modes: unsigned
(`bits`), signed (`bits ^ signbit`) and float (`bits ^ signbit` when the sign
bit is clear, `~bits` when it is set). Descending complements the mapped key,
which reverses the value order and leaves equal keys in input order, so
`argsort(x, descending=True)` equals `argsort(-x)` for every value including
`±0.0`. Test `test_descending_int64_matches_python_arithmetic` exists because
`np.argsort(-x)` is itself wrong for `INT64_MIN`: NumPy's negation wraps.

## Install and build

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build      # -> dist/libmojo-argsort.so
pixi run test
pixi run bench
```

In this shared environment, source `/nvme0n1-disk/mojo-toolchain/activate.sh`
first and then:

```bash
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

## Performance

Best-of-five wall clock, same process, every case gated on an exact
correctness check before timing. The reference column is the fastest reasonable
NumPy formulation.

| case | reference | mojo-argsort | result |
| --- | ---: | ---: | ---: |
| argsort int32 n=1048576 | 290.92 ms | 116.87 ms | 2.49x faster |
| argsort int32 n=1048576, vs `np.argsort` default quicksort | 144.60 ms | 100.63 ms | 1.44x faster |
| argsort int64 n=1048576 | 209.78 ms | 488.38 ms | **0.43x, slower** |
| argsort float64 n=1048576 | 256.68 ms | 355.08 ms | **0.72x, slower** |
| argsort int64 descending n=1048576 | 264.22 ms | 254.53 ms | 1.04x |
| argsort n=32, 200000 calls | 2418.36 ms | 10673.07 ms | **0.23x, slower** |
| counting_argsort n=1048576 b=4096 | 190.56 ms | 54.32 ms | 3.51x faster |
| lexsort 2 keys n=262144 | 81.55 ms | 100.66 ms | **0.81x, slower** |
| count_descents n=4194304 | 29.85 ms | 7.06 ms | 4.23x faster |

The wins and the losses are both real:

- 32-bit keys win because a radix sort of 4 bytes needs 4 passes, and NumPy's
  stable path is a merge sort that moves indices rather than bytes.
- 64-bit and float64 keys lose. NumPy's own integer radix sort is
  well-vectorised and beats this straightforward 8-pass implementation; at 64
  bits the pass count doubles and the extra memory traffic dominates.
- The short-input case loses to allocation and ctypes overhead, not to the
  kernel: at n=32 the sort itself is under a microsecond, while the shim
  allocates three arrays and crosses the FFI. Below the dispatch threshold the
  comparison sort is still the right algorithm, it is just dwarfed by the call.
- `lexsort` is 0.81x because it is a chain of full radix sorts, one per key,
  while `np.lexsort` walks a single `void*` argument record.
- `count_descents` wins because it is a pure reduction with no output array;
  NumPy has to materialise a full `diff` array to answer the same question.

The kernels are memory bound and deliberately serial. Threading a radix pass
makes it slower, and the porting brief for this toolchain records the same
measurement for a 4M-element axpy.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-argsort.so`.

`python/mojo_argsort` owns every array: it normalises dtypes, allocates the
scratch buffers, and makes one call per sort. Buffers cross the C ABI as 64-bit
addresses and are reconstructed in Mojo as
`Pointer[UInt64, AnyOrigin[mut=True]]`, which avoids parametric exported
functions. A single `uint64` scratch allocation serves the 32-bit path too: the
kernel builds a `UInt32` view over the first `n * 4` bytes of the same buffer.

Each radix pass is three loops: count 256 digit buckets, turn the counts into
exclusive prefix offsets, then scatter keys and values. Passes ping-pong between
two key buffers and two value buffers in an unrolled pair, so no pass ever
aliases its own input, and the final result always lands back in the first pair.
No dynamic allocation happens inside Mojo.

## Tests

`PYTHONPATH=python python -m pytest tests -q` — 57 tests.

They are parity tests against `numpy` and Python integer arithmetic, chosen so
that a plausible kernel bug fails them: a missing radix pass (caught by
`test_high_byte_dominates`, where only the top byte separates the keys), a
non-stable pass (`test_all_equal_is_identity`), a wrong sign-bit transform
(`test_extreme_values`), an off-by-one in the scatter offsets
(`test_matches_numpy_stable` over ten lengths including the 64-element dispatch
boundary), a mis-composed permutation in the multi-key chain
(`test_lexsort_three_keys_matches_numpy`), and a missing `ascontiguousarray`
(`test_non_contiguous_input`).

## License

MIT
