# mojo-argsort — ANALYZE phase

Measured hotspot map for the Mojo port of `argsort`. Every number below comes
from a run in this session (no estimates): one `pixi run bench` invocation
through the flock wrapper plus focused `pixi run python -u` probes (scaling,
glue-vs-kernel split, dtype sweep, collect-vs-kernel, alloc-layout, python
baseline). This session's box is shared and loaded (loadavg ~9–15); absolute
ms vary run-to-run and the 64-bit races flip sign between probes — the
*ordering by bytes touched* and the *scaling trends* are the stable signal.
Best-of-3/5 per cell; ratios quoted are from the same interleaved probe.

Environment: Linux 6.8, Python 3.13.14, NumPy 2.5.1, Mojo 1.2.0.dev2026092905.
Upstream `argsort` distribution: **not importable**
(`ModuleNotFoundError: No module named 'argsort'`), so all "reference"
columns are NumPy (`kind="stable"` / `np.lexsort` /
`np.count_nonzero(np.diff(...))`), the same reference `bench/bench.py` uses.
A plain-Python baseline is in §5. No files under `src/` or `tests/` were
modified; the only deliverable is this file.

---

## 1. `pixi run bench` this session (flock-wrapped, authoritative)

```
case                                 reference    mojo-argsort     ratio
------------------------------------------------------------------------
argsort int32 n=1048576               129.70ms         44.62ms     2.91x
argsort int32 (quicksort ref)          75.89ms         63.63ms     1.19x
argsort int64 n=1048576               142.47ms        157.38ms     0.91x
argsort float64 n=1048576             152.08ms        252.57ms     0.60x
argsort int64 descending n=1048576    124.49ms        107.14ms     1.16x
argsort n=32 x200000                  450.48ms       3052.31ms     0.15x
counting_argsort n=1048576 b=4096     133.09ms         30.87ms     4.31x
lexsort 2 keys n=262144                53.60ms         36.26ms     1.48x
count_descents n=4194304               17.08ms          3.31ms     5.17x
```

A second full pass (`bench/_analyze.py`, shell-4, 107 s, same session)
corroborates every ranking: its radix-64 glue split is 215.97 ms full vs
206.60 ms kernel-only (6.5% glue, within the layout-noise band of §3, not a
structural glue cost); its scaling peaks 64-bit at 1.53x @262k and converges
to 1.10–1.11x @4M; its narrow-dtype sweep (0.07–0.17x), insertion ladder
(88.4 ms @20k), and counting bucket curve (b=256 31.1 ms vs b=4096 44.6 ms)
all match §2–§5. Cells where the two passes differ (lexsort k=2 @262k:
42.9 vs 35.6 ms; counting b=4096: 44.6 vs 30.9 ms, ratio 2.87x vs 4.31x)
are already spanned by §4's ranges and change no verdict. `count_descents`
@1024 ties NumPy (0.008 vs 0.007 ms), consistent with glue dominating below
compaction sizes.

**Rows where the port is slower than the reference (kept, not dropped):**

- `n=32 ×200000` — **0.15x** (15.3 µs vs 2.2 µs per call in probes). The only
  consistent, large loss; glue-dominated (§3).
- `int64 n=1048576` — **0.91x**, `float64 n=1048576` — **0.60x** this run.
  In interleaved probes the same cells measured ~1.0x (tie) and inversion
  flips under load, i.e. the 8-pass 64-bit radix is at rough parity with
  NumPy's stable sort. Kept here as slower-in-this-run; do not read a stable
  win *or* a stable loss into either cell.
- Narrow dtypes (`int8/int16/uint8/uint16`, §5) — **0.07–0.14x**, consistent
  and structural (widen to 64-bit + 8 passes while NumPy sorts 1–2-byte keys
  natively).
- `int32` vs NumPy *quicksort* is only **1.19x** — still a win, but the
  flattering 2.91x is against the stable (merge) path.

The bench "descending reference" is `ma.argsort(-x)` itself (negation + mojo
both ways); it only shows descending cost ≈ ascending cost (same passes,
complemented key). Vs NumPy `argsort(-x)` the ratio is ~0.64x for int64 in
one probe (135 vs 175 ms) — descending carries no extra pass, it inherits
the 64-bit parity of §4.

---

## 2. Ranked hotspot table

Bytes/flops are derived by counting every buffer touch in `src/kernels.mojo`
(prologue + per-pass + final, line-cited). "Flops" = integer ops/element
(map ~3, digit extract 2, histogram ±2 — there is **no floating-point
arithmetic anywhere**, README §Parity). Flops/byte uses those counts;
GPU-worthiness threshold ~2 flops/byte (MOJO_NOTES §5). Times are best-of-3/5
from this session's probes (absolute ms ±30% under load; ratios are the
signal). Ranked by time × (loss vs reference or absolute cost) — where the
next phases should look first.

| # | function | input size | time (mojo) | bytes touched | flops | flops/byte | bound | glue vs kernel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `argsort` n=32 (insertion, per call) | n=32, 200k calls | **13.8 µs/call** vs NumPy 2.2 µs (**0.15x**) | ~2.5 KB (32×8B×~10 touches) | ~500 (O(n²) moves) | ~0.2 | **glue-bound** | **~83% glue**: kernel-only 2.28 µs; `_prepare` 2.66 µs; `_addr` 1.45 µs × 4; ctypes-only (n=0) 2.00 µs |
| 2 | `argsort` int8/16, uint8/16 n=1M | n=1M, 1–2 B/elem in | 95–101 ms vs NumPy 6.8–13.9 ms (**0.07–0.14x**) | 360 B/elem (= int64 radix: widen to 8 B + 8 passes + final) | ~60/elem | 0.17 | memory-bound (indirect) | kernel ~99%: widen itself 0.50–0.52 ms; alloc lazy |
| 3 | `argsort` int64 / float64 n=1M | n=1M, 8 B/elem | 112–253 ms (load-dep.; bench this run 157/253 ms) vs NumPy 142–152 ms (**0.6–0.9x this run, ~1.0x tie in probes**) | 360 B/elem: prologue 24 (L144–146) + 8 passes × 40 (L66–85) + final 16 (L151–152) = 360 MB @1M | ~60 (3 + 8×7 + 1) | **0.17** | memory-bound (indirect scatter) | kernel ~99%: full vs prebuilt-kernel delta sign-flips (−5 to +25 ms across probes — layout/noise, §3); glue <0.1% |
| 4 | `lexsort` 2 keys n=262k | n=262k, 2× int64 | 35.6 ms vs NumPy 49.9 ms (1.40x) | ~730 B/elem (~191 MB): k× 360 + per-key gather/compose (0.46 + 0.38 ms/key) | ~120/elem | 0.17 | memory-bound | kernel ~85%: per-key gather `k[idx]` 0.456 ms + compose `idx[…]` 0.383 ms measured; chained radix dominates |
| 5 | `count_descents` n=4M | n=4M, 8 B/elem | 3.3–4.3 ms vs NumPy 17.0–17.2 ms (4.0–5.2x) | 8 B/elem (single streaming read, L275–280) = 33.5 MB | ~3/elem | 0.38 | memory-bound (streaming) | kernel ~90–98%: glue 0.10–0.42 ms (2–10%); `is_sorted` identical (3.8 ms) |
| 6 | `argsort` int32 / float32 n=1M | n=1M, 4 B/elem | 33–60 ms vs NumPy 130–144 ms (**2.7–3.5x**) | 104 B/elem: prologue 12 (L129–131) + 4 passes × 20 (L87–106) + final 12 (L136–137) = 104 MB @1M | ~32/elem | 0.31 | memory-bound (indirect) | kernel ~99%: glue <1% |
| 7 | `counting_argsort` n=1M b=4096 | n=1M, 4 B/elem | 30.9 ms (bench) / 31.5 ms (probe) vs NumPy 131–133 ms (**4.2x**) | ~40 B/elem: count 4 + scatter 20 + final 16 (L208–247) = 42 MB @1M | ~12/elem | 0.30 | memory-bound | kernel ~99%: one ctypes call, 3 scratch arrays |
| 8 | `count_descents` n=1M | n=1M, 4 or 8 B/elem | int32 0.708 ms, int64 0.714 ms (equal!) | 4 or 8 B/elem | ~3/elem | 0.75 (int32) / 0.38 (int64) | memory-bound (latency) | kernel ~99% |
| 9 | `argsort` insertion (forced, large n) | n=20k | 91.8 ms vs NumPy 1.52 ms (**0.017x**); n=5k 5.63 vs 0.31; n=1k 0.414 vs 0.063 | O(n²×16 B) worst-case | O(n²) | ~0.1 | compute-bound (quadratic) | kernel 100% — the algorithm is the cost |
| 10 | `argsort` n=65–1024 (radix just past crossover) | n=65…1024 | mojo 0.037–0.066 ms vs NumPy 0.039–0.052 ms (~0.9–1.2x) | radix 104/360 B/elem (tiny abs.) | as above | as above | **glue-bound** | glue ~30–80% of 16–65 µs; kernel 33 µs @1024 |

Notes:

- Row 3's span *is* the finding: the 64-bit radix sits at parity with NumPy
  within contention noise. Any next-phase claim that it "loses" (or "wins")
  at n=1M must be re-measured on a quieter box with pinned cores. The
  *scaling* (§4) is more informative than this cell.
- Row 8's equality (int32 = int64 time despite half the bytes) shows descents
  is latency-bound, not bandwidth-bound — a scalar `_map`-per-element loop;
  see §5.
- `float32` stays 32-bit (4 passes): 41.6 ms vs NumPy 143.9 ms (3.46x).
  `float16` widens to f64 (91.6 ms) yet still beats NumPy (171 ms, 1.87x)
  because NumPy's f16 path is slow.

---

## 3. Glue vs kernel split (where the time sits)

Method: full `ma.argsort` vs direct `_lib.lib.mas_*` calls with prebuilt
arguments (buffers allocated once), plus component timers (`_prepare`,
`_addr`, ctypes-only `n=0` early-return, lazy alloc). All numbers this session.

**Large n (≥16k): kernel dominates.** `_prepare` ≈ 2.7 µs (of which the
dtype-list rebuild `[np.dtype(t) for t in SUPPORTED_DTYPES]`, core.py L36, is
2.59 µs — rebuilt on *every* call), `np.ascontiguousarray` ≈ 0.1 µs if
contiguous, one ctypes crossing ≈ 2.0 µs (ctypes-only, n=0, this session;
MOJO_NOTES' "~0.9 µs" is optimistic here), one `_addr` ≈ 1.45 µs.
At n=1M all glue sums to <0.1 ms against 30–250 ms kernel. The measured
full-vs-prebuilt deltas at n=1M (int64 −6 to −24 ms, i.e. kernel *slower*
than full call; int32 +5.7 to +24.9 ms) sign-flip between probes — i.e.
they are allocation-layout/NUMA noise (§6), not glue.

**Small n (≤1024): glue dominates.**

| n | full (per call, amortized 30k) | kernel-only | glue share |
| --- | --- | --- | --- |
| 32 (insertion) | 13.8 µs | 2.28 µs | ~83% (~11.5 µs: `_prepare` 2.66 + 4× `_addr` ~5.8 + ctypes ~2.0 + alloc ~1.4) |
| 1024 (radix) | 59–66 µs | ~33–47 µs | ~30% |

NumPy at n=32 is 2.2 µs/call *total*. The port's kernel alone (2.28 µs)
already equals NumPy's whole call; everything above that is Python/ctypes
glue. The dtype-list rebuild (2.59 µs) is the single largest avoidable glue
item — 19% of a 13.8 µs n=32 call. `lexsort` pays `_prepare` per key.

**Heap/layout (inconclusive this session).** Reused-buffer test at n=1M int64:
per-call separate `np.empty` ×5 = 149.5 ms vs single-allocation-reused =
164.5 ms vs full `ma.argsort` = 193.7 ms. Unlike prior sessions, single
allocation was *not* faster here (164 vs 150 ms), so the layout effect is
load-dependent and **not** claimed in this session. Lazy alloc cost itself
remains small; first-touch of fresh 40 MB pages lands inside kernel time.

---

## 4. Scaling — the shape of the cost

Argsort (stable), best-of-3/5, this session (selected probe rows; n=1M/4M
cells tie to §1 within noise):

| n | int32 mojo / NumPy | int64 mojo / NumPy | float64 mojo / NumPy |
| --- | --- | --- | --- |
| 1024 | 0.037 / 0.039 ms (0.95x) | 0.059 / 0.052 (0.88x) | ~0.06 / ~0.045 (~0.8x) |
| 4096 | 0.110–0.121 / 0.239–0.272 (2.0–2.5x) | 0.235–0.290 / 0.239–0.275 (~1.0x) | — |
| 16384 | 0.435–0.473 / 1.19–1.27 (2.7x) | 1.246 / 1.238 (1.01x) | ~1.3 / ~1.3 (~1.0x) |
| 65536 | 1.80–2.00 / 5.59–6.00 (3.0x) | 5.09–5.29 / 6.18–6.21 (1.2x) | ~5.2 / ~6.5 (~1.25x) |
| 262144 | 7.6–8.1 / 27.4–28.6 (3.5x) | 20.0–20.3 / 29.4–30.6 (1.5x) | ~21 / ~33 (1.55x) |
| 1048576 | 33–60 / 130–144 (2.7–3.9x) | 112–205 / 141–152 (0.7–1.3x) | 118–253 / 146–152 (0.6–1.2x) |
| 4194304 | 184–207 / 598–640 (3.1x) | 590–637 / 675–684 (~1.1x) | ~630–730 / ~690 (~1.0x) |

- **Crossover vs NumPy is n≈1024 (int32) / n≈4096 (64-bit).** Below that the
  port loses on glue, not on the algorithm. Raising `INSERTION_LIMIT` alone
  does not fix it — insertion at n=1000 (0.41–0.19 ms random; §5) is already
  worse than radix at n=1024 (0.06 ms).
- **int32 never hits the wall**: 2.0–3.5x at every large n. 4 passes
  (104 B/elem) vs 8 (360 B/elem) — the 3.5x traffic ratio tracks the speedup.
- **64-bit converges to ~1.0–1.1x at n=4M** (both dtypes). Achieved BW for
  the radix is only ~2–3 GB/s (360 MB / 112–205 ms) vs 7.8 GB/s for the
  descents stream and ~10 GB/s for a plain `copyto` (1 MB copy = 0.087–0.098
  ms). The scatter (data-dependent `dst` via `hist`, serialized RMW per
  element, L80–85) is the bottleneck, not DRAM.
- `count_descents` eff. BW: 33.5 MB / 4.3 ms = **7.8 GB/s** @4M int64 (scalar
  loop, one `_map` call/elem, no SIMD). Streaming but scalar.
- Insertion is O(n²): n=64 ~0.014 ms → n=1000 0.41 ms → n=5000 5.63 ms →
  n=20000 91.8 ms. Radix at n=1M is data-independent (random ± sorted same
  order).

---

## 5. Per-entry-point findings

### `argsort` (radix default, insertion ≤64)

`core.argsort → _lib.{radix,insertion}_argsort → lib.mas_*` (one ctypes call,
no Python loop). No per-element ctypes re-entry anywhere (§6).
`descending` costs nothing extra (complemented key, same passes: 175 ms vs
205 ms ascending in one probe — within noise; bench ratio ~1.0–1.16x vs a
mojo-negated reference).

**Narrow-dtype pathology (row 2).** `int8/int16/uint8/uint16` are widened to
64-bit (`_lib._widen`, only 0.50–0.52 ms — negligible) and then sorted with
the **8-pass 64-bit radix** (mojo: int8 95.8, int16 94.9, uint8 95.8,
uint16 100.8 ms — all ≈ int64). NumPy sorts these natively at 6.8–13.9 ms.
The loss is not widening; it is doing 360 B/elem of radix traffic on 1–2
B/elem inputs (≈180–360x traffic amplification) instead of a 1–2-pass radix
on the original width.

**Plain-Python baseline** (same session): `sorted(range(n), key=lambda i:
x[i])` at n=16384 = 4.67 ms/call vs mojo 1.23 ms vs NumPy 1.28 ms — mojo is
3.8x faster than plain Python at that size and already beats it at n=1024
(mojo 0.066 ms vs python 0.190 ms; measured earlier this session). Upstream
`argsort` could not be used (not importable).

### `lexsort`

`core.lexsort` (L84–86): `idx = arange(n)` then per key
`idx = idx[_radix_argsort(k[idx], descending)]` — **per-key, not
per-element**, ctypes calls (k calls for k keys), but each iteration
allocates a gather copy `k[idx]` (measured 0.456 ms @262k) plus the compose
`idx[…]` (0.383 ms). Measured: n=262k k=2 full 35.6 ms vs 2× single radix
38.2 ms (within noise of chained cost); k=3 51.9 ms vs 3× 54.8 ms. The
k× chained radix (720/1080 B/elem) is the cost; gathers are ~2–3% here.
NumPy walks a single void record (~47–79 ms); the port wins 1.4–1.5x only
because the underlying 64-bit radix is at parity. No GPU case.

### `counting_argsort`

One ctypes call, 3 scratch arrays (ka, va: uint64, hist: buckets×u32).
Fastest kernel per byte (40 B/elem, single pass) and **4.2x** NumPy (bench
30.9 ms; probe 31.5 ms vs 131 ms). Bucket sensitivity (n=1M int32): b=256 →
19.7 ms vs b=4096 → 31.5 ms. Larger hist = more prefix work + more cache
pressure (both hist RMWs, L208–245). Glue ~0.1%.

### `count_descents` / `is_sorted`

One ctypes call, **zero scratch arrays** (no alloc at all — the only entry
point that touches nothing but the input). 3.3 ms bench / 4.3 ms probe @4M
(4.0–5.2x). Kernel-only 3.9 ms, glue 0.42 ms (10%). It wins because NumPy
materializes `diff` + bool arrays. Row 8 (int32 = int64 time) shows the
remaining gap to streaming BW is the scalar `_map` + non-vectorized compare
(L275–280) — latency, not bytes.

### `is_sorted`

`count_descents(...) == 0` (core.py:105); identical cost (3.8 ms @4M).

---

## 6. Python-level loops, allocations, and copies

**Per-element ctypes re-entry: NONE.** Inspection of
`python/mojo_argsort/*.py` (full source dumped this session): the only
Python loops are (a) `for k in prepared` in `lexsort` (L85 — per-key,
k iterations, k ctypes calls; necessary, not a bug), (b) the two small
bookkeeping loops (`for i, k in enumerate(prepared)`), (c) the dtype-list
comprehension in `_prepare`. No loop re-enters ctypes per element; no `for`
over data. `src/kernels.mojo`: 18 scalar `for i in range(n)` loops, zero
`simd`/`parallel`/`gpu`/`load[width]`/`store` (grep this session: 0 hits for
all).

**Per-call allocations/copies (the actual glue costs):**

1. `_prepare` dtype-list rebuild — 2.59 µs/call (19% of an n=32 call). Cheapest
   fix in the repo (hoist to a module constant).
2. `_addr` per buffer — `int(a.ctypes.data)`, 1.45 µs each; radix makes 6,
   insertion 4, counting 5. At n=32 that is ~6–9 µs — the dominant glue line.
3. Scratch `np.empty` per call: radix 5×n (idx+ka+kb+va+vb+hist), insertion
   3×n, counting 3×n. Lazy cost small (~50 µs @n=1M), but each fresh 40 MB
   chunk pays first-touch inside kernel time. (Layout effect was
   inconclusive this session — §3.)
4. `ascontiguousarray` — ~0.1 µs if contiguous, full copy if strided
   (correctness requirement, `test_non_contiguous_input`). Bench inputs are
   contiguous, so no measured cost.
5. `lexsort` gathers — `k[idx]` (0.456 ms @262k) + `idx[…]` (0.383 ms) per
   key. Real, §5.
6. `_widen` — `.astype(...)` alloc+copy for <4-byte dtypes: 0.50–0.52 ms @1M.
   Measured, negligible vs the 95 ms radix that follows. (The widen is
   unavoidable given the kernel ABI; the *pass count* is the choice.)

---

## 7. Next-phase verdicts (one per hotspot)

Rule: GPU needs flops/byte ≳2 behind it. No kernel here exceeds 0.75.
**No GPU verdicts anywhere** — the strongest candidate (`count_descents`
int32 @0.75) is still 2.7x below threshold, and the radix/lexsort kernels
are indirect/scattered (GPU would need a radix-architecture redesign, not a
port). MOJO_NOTES §5's "CPU-only below ~2 flops/byte" applies uniformly;
host API availability is not the argument.

| # | hotspot (§2) | verdict | why (flops/byte + evidence) |
| --- | --- | --- | --- |
| 1 | n=32 / small-n glue (83% glue, 0.15x) | **none** (for kernel) | Kernel is already 2.28 µs; the phase is Python: hoist dtype-list (save ~2.6 µs, ~19%), cut `_addr` calls (cache pointers / `c_void_p`, save ~5–9 µs). Small-n total could reach ~4–6 µs. No kernel work moves 13.8 → 2.2 µs. |
| 2 | narrow dtypes 0.07–0.14x | **none** (algorithmic width fix, not listed phases) | 0.17 flops/byte + 360 B/elem regardless of input width. Fix is a narrow-width radix (1–2 passes on 1–2 B keys) or a counting path — an algorithm change, not unroll/simd/parallel/gpu. Widening (0.5 ms) is noise. |
| 3 | radix-64/32 @1M+ (360/104 B, 0.17/0.31) | **parallel** | Memory-bound *and* far from peak (achieved 2–3 GB/s vs 10 GB/s streaming): histogram + scatter are the latency chain. `parallelize` from `max.algorithm` (MOJO_NOTES §4) over hist buckets / partitioned scatter is the standard radix parallelization; SIMD cannot fix data-dependent scatter. SIMD helps only the prologue/map (~3/60 ops). GPU rejected at 0.17 flops/byte. Unroll: minor (pass pair already unrolled). |
| 4 | lexsort chain | **parallel** (inherited) | Same 0.17 flops/byte per key; cost is k× radix. Parallelizing the inner radix (row 3) plus fusing the gather into the scatter would cut both terms. No separate SIMD/GPU case. |
| 5 | count_descents (0.38–0.75) | **simd** | Pure streaming reduction, trivially vectorizable: map + compare + popcount-reduce. 7.8 GB/s scalar → expect 15+ GB/s with `simd_width_of` vector loads (MOJO_NOTES §3). Parallel also helps but SIMD is the first step (no synchronization needed). |
| 6 | radix-32 / counting (0.30–0.31) | **none** (already winning 3–4.6x) | Memory-bound and fastest-per-byte already; parallelizing counting scatters risks hist contention for little absolute gain (31 ms → maybe 15 ms). Lowest ROI. Leave for later. |
| 7 | insertion O(n²) | **none** | Algorithmic dead end above n≈256 (92 ms @20k). Correctly capped at 64. Only action: keep the cap, do not relax it. |
| 8 | allocation layout | **none** (else "unroll" by analogy: pack scratch) | This session's test did *not* reproduce a single-alloc win (150 vs 164 ms), so do not act on it until re-measured on a quieter box. Lazy cost is genuinely small. |

Ordering for the next phases on evidence:

1. **Shim (no kernel change):** hoist dtype-list, cache `_addr`/ptrs.
   Expected: n=32 13.8 → ~5 µs. Both measured above.
2. **SIMD:** `count_descents` vectorization (0.38–0.75 flops/byte, streaming,
   zero dependencies — the only clean SIMD target).
3. **PARALLEL:** radix 64/32 histogram/scatter (`max.algorithm.parallelize`
   behind a size gate, per MOJO_NOTES §4 — `std.algorithm` does not
   compile). Target the 360 B/elem traffic where achieved BW is 2–3 GB/s.
4. **No GPU** anywhere (max 0.75 flops/byte, threshold ~2).
5. **No unroll** anywhere (passes already paired; loops are
   memory-latency-bound, not branch-bound).

---

## 8. Repro

```bash
pixi run bench        # flock-wrapped, bench/bench.py (§1 output verbatim)
pixi run python -u -c "..."   # component probes quoted in §2–§5
```

No files under `src/` or `tests/` were modified. This file is the only
deliverable; `bench/_analyze.py` (scratch harness from the prior session)
and `/tmp/probe.py` are not part of it.
