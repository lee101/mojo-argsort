# mojo-argsort — ANALYZE phase

Measured hotspot map for the Mojo port of `argsort`. Everything numerical
below comes from runs in this session (no estimates): three `pixi run bench`
invocations through the flock wrapper plus focused `pixi run python` probes
(scaling sweeps, glue-vs-kernel splits, dtype sweeps, layout experiments).
Raw harness: `bench/_analyze.py` (temporary probe, not part of the deliverable).

Environment: Linux 6.8, 72× Xeon E5-2697 v4 (2 sockets × 18 cores × 2 HT),
251 GiB RAM, Python 3.13.14, NumPy 2.5.1, Mojo 1.2.0.dev2026092905.
Upstream `argsort` distribution: **not importable** (`ModuleNotFoundError`),
so all "reference" columns are NumPy (`kind="stable"` / `np.lexsort` /
`np.count_nonzero(np.diff(...))`), same as `bench/bench.py`.

⚠️ **Shared-box contention caveat (affects every ms below).** The box is
heavily loaded by unrelated tenants (loadavg 5–54 during this session;
`ps` shows multi-thousand-percent workloads). Hence absolute ms vary
±30–50% run-to-run, and Numpy-vs-Mojo races on the 64-bit paths actually
**flip sign between runs** (see §1). What is stable across runs is the
*ordering by bytes touched* and the *scaling trends*; single-point ratios
near 1.0× should be read as "tied within noise". Best-of-N with
interleaving was used throughout; where min/median disagree materially the
spread is quoted.

---

## 1. Authoritative `bench/bench.py` results (three flock-wrapped runs, same session)

| case | run 1 (load ~6) | run 2 (load ~43–54) | run 3 (load ~43) | final probe (load ~9) |
| --- | --- | --- | --- | --- |
| argsort int32 n=1048576 (vs stable) | 130.60 → 37.43 ms (3.49×) | 149.97 → 55.16 ms (2.72×) | 126.47 → 37.42 ms (3.38×) | 132.00 → 33.63 ms (3.92×) |
| int32 vs quicksort | 70.89 → 38.71 (1.83×) | 89.17 → 52.94 (1.68×) | 67.91 → 37.52 (1.81×) | — |
| argsort int64 n=1048576 | 134.06 → 117.43 (1.14×) | 165.30 → 209.56 (**0.79×, slower**) | 134.79 → 122.24 (1.10×) | 146.47 → 147.59 (0.99×, tie) |
| argsort float64 n=1048576 | 149.14 → 121.03 (1.23×) | 184.30 → 286.51 (**0.64×, slower**) | 150.03 → 191.89 (**0.78×, slower**) | 168.36 → 143.56 (1.17×) |
| int64 descending | 108.05 → 95.19 (1.14×*) | 150.25 → 147.57 (1.02×*) | 110.31 → 96.39 (1.14×*) | — |
| argsort n=32 ×200k | 456.21 → 2995.57 (**0.15×**) | 522.44 → 2863.78 (**0.18×**) | 424.14 → 3056.73 (**0.14×**) | 15.3 µs vs 2.2 µs/call (**0.14×**) |
| counting n=1M b=4096 | 130.01 → 30.54 (4.26×) | 124.56 → 29.83 (4.18×) | 136.90 → 31.45 (4.35×) | 125.02 → 27.11 (4.61×) |
| lexsort 2 keys n=262144 | 52.08 → 36.64 (1.42×) | 48.58 → 37.51 (1.30×) | 54.82 → 36.97 (1.48×) | 47.10 → 28.10 (1.68×) |
| count_descents n=4M | 17.25 → 2.92 (5.90×) | 16.02 → 3.56 (4.50×) | 17.81 → 3.22 (5.54×) | 14.13 → 2.42 (5.84×) |

\* bench's "descending reference" is `ma.argsort(-x)` itself, not NumPy —
it times negation + mojo both ways, so the ratio only shows descending
cost ≈ ascending cost (no extra pass). Vs NumPy `argsort(-x)` the ratio
is ~0.96–4.27× depending on dtype (int64 ties, int32 wins 4.27×).

**Rows where the port is slower than upstream/NumPy (kept, not dropped):**

- `n=32 ×200k` — **0.14–0.18× in all runs**, the only consistent, large
  loss. Glue-dominated (see §3).
- `int64 n=1M` and `float64 n=1M` — **flipping** (1.14×/1.23× in run 1,
  0.79×/0.64× in run 2, tie in the final probe). Within contention noise
  of 1.0×: the 8-pass 64-bit radix is at rough parity with NumPy's stable
  sort, not a stable win and not a stable loss. README's older table
  (0.43×/0.72×) is stale vs this session's re-measurement.
- Narrow dtypes (`int8/int16/uint8/uint16`, §5) — **0.07–0.23×**,
  consistent, structural (widen to 64-bit + 8 passes while NumPy sorts
  1–2-byte keys natively).

---

## 2. Ranked hotspot table

Bytes/flops derived by counting every buffer touch in `src/kernels.mojo`
(prologue + per-pass + final, lines cited). "Flops" = integer ops/element
(map ±3, digit extract 2, hist ±2 — there is no floating-point arithmetic
anywhere, README §Parity). Flops/byte uses those counts; threshold for
GPU-worthiness ~2 (MOJO_NOTES §5). Times are best-of-5, n=1M unless noted,
from this session's probes (absolute ms ±30% under load; ratios are the
signal).

Ranked by **time × (loss vs reference or absolute ms)** — i.e. where the
next phases should look first:

| # | function | input size | time (mojo) | bytes touched | flops (est. from code) | flops/byte | bound | glue vs kernel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `argsort` n=32 (insertion, per call) | n=32, 200k calls | **15.3 µs/call** vs NumPy 2.2 µs (**0.14×**) | ~2.5 KB (32×8×~10) | ~500 (O(n²) moves) | ~0.2 | **glue-bound** | **~80% glue**: kernel-only 2.2 µs; `_prepare` 2.7 µs; alloc ~1.4 µs; 5× `_addr` + ctypes ~8–10 µs |
| 2 | `argsort` int8/16, uint8/16 n=1M | n=1M, 1–2 B/elem in | 85–105 ms vs NumPy 7–19 ms (**0.07–0.23×**) | 360 B/elem (= int64 radix: widen to 8 B + 8 passes + final) | ~60/elem | 0.17 | memory-bound (indirect) | kernel ~99%: widen itself 0.5 ms; 5-way alloc lazy 0.05 ms |
| 3 | `argsort` int64 / float64 n=1M | n=1M, 8 B/elem | 90–210 ms (load-dep.) vs NumPy 130–185 ms (~**1.0×, tie**) | 360 B/elem: prologue 24 (L144–146) + 8 passes ×40 (L66–85) + final 16 (L151–152) = 360 MB @1M | ~60 (3 + 8×7 + 1) | **0.17** | memory-bound (indirect scatter) | kernel ~99%: `_prepare` 3 µs, alloc 0.05 ms, ctypes 5 µs — all <0.01% |
| 4 | `lexsort` 2 keys n=262k | n=262k, 2× int64 | 28–37 ms vs NumPy 47–55 ms (1.3–1.7×) | ~2×360 + gathers ≈ 730 B/elem (~191 MB) | ~120/elem | 0.17 | memory-bound | kernel ~85%: per-key gather+compose +1.9–5 ms (+14%): `a[idx]` 0.8 ms + `idx[…]` 1.9 ms |
| 5 | `count_descents` n=4M | n=4M, 8 B/elem | 2.4–3.6 ms vs NumPy 14–18 ms (4.5–5.9×) | 8 B/elem (single streaming read, L275–280) = 33.5 MB | ~3/elem | 0.38 | memory-bound (streaming) | kernel ~95–98%: glue 0.01–0.18 ms (1–5%) |
| 6 | `argsort` int32 / float32 n=1M | n=1M, 4 B/elem | 34–55 ms vs NumPy 126–150 ms (**2.7–3.9×**) | 104 B/elem: prologue 12 (L129–131) + 4 passes ×20 (L87–106) + final 12 (L136–137) = 109 MB @1M | ~32/elem | 0.31 | memory-bound (indirect) | kernel ~99%: glue <0.01% |
| 7 | `counting_argsort` n=1M b=4096 | n=1M, 4 B/elem | 27–39 ms vs NumPy 125–137 ms (**4.2–4.6×**) | ~40 B/elem: count 4 + scatter 20 + final 16 (L208–247) = 42 MB @1M | ~12/elem | 0.30 | memory-bound | kernel ~99.9%: glue +0.05 ms (+0.1%) |
| 8 | `count_descents` n=1M int32 | n=1M, 4 B/elem | 1.41 ms (vs int64 0.79 ms — wider is faster here!) | 4 B/elem | ~3/elem | 0.75 | memory-bound | kernel ~99% |
| 9 | `argsort` insertion (large n) | n=20k forced | 86.6 ms vs NumPy 1.25 ms (**0.01×**); n=5k: 4.8 vs 0.24 ms | O(n²×16 B) worst-case | O(n²) | ~0.1 | compute-bound (quadratic) | kernel 100% — algorithm is the cost |
| 10 | `argsort` n=65–1024 (radix, just past crossover) | n=65…1024 | 23–92 µs vs NumPy 2.3–64 µs (0.10–0.70×) | radix64/32 as above (tiny) | as above | as above | **glue-bound** | glue 30–64%: n=256: glue +18.6 µs (+64%); n=1024: +14.5 µs (+30%) |

Notes on the table:

- Row 3's span (90–210 ms) *is* the finding: the 64-bit radix is at
  parity with NumPy within contention noise. Any next-phase claim that it
  "loses" (or "wins") at n=1M must be re-measured on a quieter box with
  pinned cores. The *scaling* (§4) is more informative than this cell.
- Row 8's inversion (int32 *slower* than int64 for a streaming read) is
  real and load-independent: the 32-bit path reloads 4 B/elem but the
  scalar `_map32` + loop overhead per element dominates, so fewer bytes
  ≠ less time. It reinforces that descents is latency-bound, not
  bandwidth-bound (§4).
- `is_sorted` = `count_descents == 0` (core.py L105), identical timing
  (2.9 ms @4M), no separate row.

---

## 3. Glue vs kernel split (how the time is divided)

Method: full `ma.argsort` vs `_lib` direct vs prebuilt-argument
`lib.mas_*` call (buffers allocated once), plus component timers (`_prepare`,
`_addr`, alloc-lazy, ctypes-only `n=0` early-return).

**Large n (≥16k): kernel ≥96–99.9%.** `_prepare` ≈ 2.9 µs (of which the
dtype-list rebuild `[np.dtype(t) for t in SUPPORTED_DTYPES]` in
`core.py` L36 is ~2.3–2.8 µs — rebuilt on *every* call), `np.ascontiguousarray`
≈ 0.1 µs if already contiguous, 5-way `np.empty(n=1M)` lazy ≈ 49 µs
(0.05% of 100 ms), one ctypes crossing ≈ 5.0 µs, one `_addr` ≈ 1.8–2.3 µs.
At n=1M all glue sums to <0.05 ms against 30–200 ms kernel: **unmeasurable**
at this scale. The measured full-vs-kernel deltas at n=1M (±15–70 ms,
sign-flipping in ABA runs) are allocation-*layout*/NUMA noise (§6), not
glue — proved by the fact that lazy alloc itself is 0.05 ms.

**Small n (≤1024): glue dominates.** Amortized (timeit, N=30k):

| n | full | kernel-only (prebuilt) | glue share |
| --- | --- | --- | --- |
| 32 (insertion) | 15.3 µs | 2.2 µs | **~80%** (≈13 µs: `_prepare` 2.7 + alloc 1.4 + 5× `_addr`/ctypes ~8) |
| 64 (insertion) | 15.7 µs | 3.2 µs | ~80% |
| 256 (radix) | 29.0 µs | 10.4 µs | ~64% |
| 1024 (radix) | 47.8 µs | 33.3 µs | ~30% |
| 4096 (radix) | 239 µs | 185 µs | ~23% (inflated by load spike) |
| 16384 (radix) | 1176 µs | 1135 µs | ~3.4% |
| 65536+ | — | — | <2% |

NumPy at n=32 is 2.2 µs/call total. The port's kernel alone (2.2 µs) already
*equals* NumPy's whole call; everything above that is Python/ctypes glue.
MOJO_NOTES' "~0.9 µs per ctypes call" is optimistic for this box/lib:
measured ctypes-only (n=0 early return) is 5.0 µs including one `_addr`.

**Dtype-list rebuild** (`core.py` L36, `[np.dtype(t) for t in …]` per
`_prepare` call, ~2.3–2.8 µs/call) is the single largest *avoidable*
glue item at small n — 12–18% of a 15 µs n=32 call. Two `_prepare` calls
are not extra (1 per public call), but `lexsort` calls it per key.

**`_addr` cost**: `int(a.ctypes.data)` ≈ 1.8–2.3 µs per call (the
`ctypes.data` property allocates). `radix_argsort` makes 6, `insertion`
4, `counting` 5. At n=32 that is ~7–10 µs — the dominant glue line.
`__array_interface__` is not faster (1.97 µs); caching the address
expression or passing `ctypes` pointers would be the fix (next phase).

**Allocation layout (production path is the slow layout).** `_lib.radix_argsort`
allocates five separate `np.empty(n)` (ka, kb, va, vb, idx) — back-to-back
8 MB mmaps at page-aligned stride 8 MB. Reused-buffer experiments at n=1M
int64 (clean, ABA-interleaved, correctness-checked):

- separate, back-to-back (production): ~187–216 ms
- single 40 MB allocation, 5 contiguous slices: ~137–185 ms (**~40 ms faster**)
- single allocation with 4 KB gaps: ~123–177 ms (not robust across rounds)
- full fresh-alloc path per call: ~189–226 ms

int32 shows the same direction (~5–10 ms on ~40 ms). The effect is real
but load-modulated; the honest claim is **single-allocation is
consistently faster by ~15–25% for the 64-bit radix at n=1M**, never
slower in any round. Note fresh-per-call allocs do *not* each pay 40 MB
of page faults in the kernel (lazy alloc is 0.05 ms); the layout itself
(cache-set aliasing between the five 8 MB-strided streams) is the
mechanism. See §6.

---

## 4. Scaling — the shape of the cost (more reliable than any single cell)

Argsort (stable), best-of-5 per n, from this session:

| n | int32 mojo / NumPy | int64 mojo / NumPy | float64 mojo / NumPy |
| --- | --- | --- | --- |
| 32 | 16.0 / 2.0 µs (0.13×) | 16.5 / 2.3 µs (0.14×) | 17.4 / 2.3 µs (0.13×) |
| 64 | 16.8 / 2.7 (0.16×) | 17.8 / 3.0 (0.17×) | 18.5 / 3.2 (0.17×) |
| 65 (radix kicks in) | 22.9 / 2.3 (0.10×) | 26.7 / 2.8 (0.10×) | 27.7 / 3.0 (0.11×) |
| 128 | 25.2 / 4.2 (0.17×) | 28.5 / 4.6 (0.16×) | 29.7 / 4.8 (0.16×) |
| 256 | 28.4 / 7.3 (0.26×) | 35.4 / 7.8 (0.22×) | 34.0 / 9.0 (0.26×) |
| 1024 | 41.2 / 49.2 (1.19×) | 63.8 / 46.0 (0.72×) | 63.3 / 41.0 (0.65×) |
| 4096 | 114 / 257 (2.25×) | 251 / 265 (1.05×) | 258 / 292 (1.13×) |
| 16384 | 0.48 / 1.21 ms (2.52×) | 1.18 / 1.22 (1.03×) | 1.32 / 1.28 (0.97×) |
| 65536 | 2.25 / 5.85 (2.60×) | 5.14 / 6.18 (1.20×) | 5.21 / 6.63 (1.27×) |
| 262144 | 9.87 / 27.0 (2.73×) | 20.1 / 28.9 (1.44×) | 21.2 / 32.9 (1.55×) |
| 1048576 | 34.4 / 128.7 (3.74×) | 106.6 / 140.6 (1.32×) | 117.9 / 146.4 (1.24×) |
| 4194304 | 183 / 606 (3.31×) | 627 / 659 (1.05×) | 727 / 695 (0.96×) |

- **Crossover vs NumPy is n≈1024 (int32) / n≈4096 (64-bit)**. Below that
  the port loses on glue, not on the algorithm. Raising
  `INSERTION_LIMIT` alone does not fix it — insertion at n=1024/2048
  (253 µs / 1010 µs) is already far worse than radix (60 / 92 µs).
- **int32 never hits the wall**: 2.5–3.7× at every large n. 4 passes
  (104 B/elem) vs 8 (360 B/elem) — the 3.5× traffic ratio tracks the
  speedup.
- **64-bit converges to ~1.0× at n=4M** (both dtypes). The radix costs
  360 B/elem; at 33.5 MB×8 passes = 300 MB+ of traffic the kernel's
  achieved BW is only ~2–5 GB/s (360 MB / 110–180 ms), vs 27 GB/s for a
  plain preallocated `copyto` and 10 GB/s for `count_descents`. The
  scatter (data-dependent `dst` via `hist`, serialized RMW per element,
  L80–85) is the bottleneck, not DRAM.
- `count_descents` eff. BW: 10.5 GB/s @1M, 8.8 GB/s @4M, 8.0 GB/s @16M
  (scalar loop, one `_map` call/elem, no SIMD). Streaming but scalar —
  next phase has headroom (see verdicts).
- Insertion is O(n²): n=64 14 µs → n=1000 189 µs (random) / 19 µs
  (sorted) → n=5000 4.8 ms → n=20k 86.6 ms. Sorted-vs-random at n=1000
  (10×) proves data-dependence; radix at n=1M is data-independent
  (random 150 ms vs sorted 189 ms — the sorted case is *slower*, i.e.
  sortedness brings no radix benefit, as designed).

---

## 5. Per-entry-point findings

### `argsort` (radix default, insertion ≤64)
Four exports back it (`mas_radix_argsort`, `mas_insertion_argsort`) via
`core.argsort → _lib.{radix,insertion}_argsort → lib.mas_*` (1 ctypes
call, no Python loop). No per-element ctypes re-entry anywhere (see §7).
`descending` costs nothing extra (complemented key, same passes:
ascending 106 ms vs descending 140 ms @1M int64 in one probe, 95 ms in
bench — within noise; bench ratio ~1.0–1.14×).

**Narrow-dtype pathology (row 2).** `int8/int16/uint8/uint16` are widened
to 64-bit (`_lib._widen`, ~0.5 ms — negligible) and then sorted with the
**8-pass 64-bit radix** (ms: int8 94–98, int16 90–105, uint8 85–98,
uint16 91–96, all ≈ int64 86–95). NumPy sorts these natively at 7–19 ms.
The loss is not widening, it is doing 360 B/elem of radix traffic on
1–2 B/elem inputs (180× traffic amplification) instead of, say, a 1–2
pass radix on the original width. `float32` is fine (stays 32-bit,
4 passes, 35–38 ms, 3.6–4.0×); `float16` widens to f64 (81 ms, still
2.2× because NumPy's f16 path is slow).

### `lexsort`
`core.lexsort` (L84–86): `idx = arange(n)` then per key
`idx = idx[_radix_argsort(k[idx], descending)]` — **per-key, not
per-element** ctypes calls (k calls for k keys), but each iteration
allocates a gather copy `k[idx]` (fancy-index, 0.4–0.8 ms @262k) plus
the radix's 5 scratch arrays plus the compose `idx[…]`. Measured:
n=262k k=2 full 35–39 ms vs 2× single radix 33 ms → overhead +2–5 ms
(**+9–14%**). k=3: 66 ms vs 3× single radix (~56 ms); k× chained radix is the
dominant cost (2×/3× the radix bytes = 720/1080 B/elem). NumPy walks a
single void record (~47–77 ms); the port wins 1.2–1.7× here only
because the underlying 64-bit radix is at parity. No GPU case.

### `counting_argsort`
`_lib.counting_argsort` → 1 ctypes call, 3 scratch arrays (ka, va:
uint64, hist: buckets×u32). Fastest kernel per byte (40 B/elem, single
pass) and 4.2–4.6× NumPy. Bucket sensitivity (n=1M int32): b=256 →
15.1 ms vs b=4096 → 27–30 ms vs numpy stable ~131 ms. Larger hist =
more prefix work + more cache pressure (both hist RMWs in the count
and scatter loops, L208–245). Glue +0.1%.

### `count_descents` / `is_sorted`
1 ctypes call, **zero scratch arrays** (no alloc at all — the only
entry point that touches nothing but the input). 2.4–3.6 ms @4M int64
(5.9×). Kernel-only 2.8–3.1 ms, glue 0.09–0.18 ms. By design it cannot
lose on allocation; it wins because NumPy materializes `diff` + bool
arrays. The remaining gap to `copyto` BW (10 vs 27 GB/s) is the scalar
`_map`-per-element + non-vectorized compare (L275–280). Narrow inputs
are widened (`uint16` → uint64 path, 1.24 ms @1M — fine).

---

## 6. Python-level loops, allocations, and copies

**Per-element ctypes re-entry: NONE.** Grep of `python/mojo_argsort/*.py`:
the only Python loops are (a) `for k in prepared` in `lexsort`
(per-key, k iterations, k ctypes calls — necessary, not a bug), (b) the
dtype-list comprehension in `_prepare` (per call, 2.3–2.8 µs), (c) index
bookkeeping. No loop re-enters ctypes per element; no `for` over data.
`src/kernels.mojo` contains **no SIMD, no `parallelize`, no GPU**
(`grep simd|parallel|gpu|load\.|store` → empty): every loop is scalar
`for i in range(n)`.

**Per-call allocations/copies (the actual glue costs):**

1. `_prepare` dtype-list rebuild — `_prepare` builds
   `[np.dtype(t) for t in SUPPORTED_DTYPES]` (11 dtypes) on every call,
   ~2.3–2.8 µs. Cheapest fix in the repo (hoist to module constant).
2. `_addr` per buffer — `int(a.ctypes.data)`, ~1.8–2.3 µs × 4–6 buffers.
   At n=32 this and (1) together are ~60% of total time.
3. Scratch `np.empty` per call: radix 5×n (idx+ka+kb+va+vb+hist),
   insertion 3×n, counting 3×n. Lazy cost ~49 µs @n=1M (0.05%), ~1.4 µs
   @n=32. Not a syscall cost — numpy caches the allocator — but the
   *layout* of five separate 8 MB mmaps costs ~15–25% at n=1M (§3, and
   would vanish with one allocation). Also each fresh 40 MB chunk gets
   fresh pages whose first touch lands inside kernel time.
4. `ascontiguousarray` — ~0.1 µs if contiguous, full copy if strided
   (correctness requirement, `test_non_contiguous_input`). Bench inputs
   are contiguous, so no measured cost.
5. `lexsort` gathers — `k[idx]` per key (copy of n elements, ~0.8 ms
   @262k) + `idx[…]` compose (~1.9 ms). Real, counted in §5.
6. `_widen` — `.astype(np.int64/uint64/float64)` alloc+copy for
   <4-byte dtypes: 0.5–1.5 ms @1M. Measured, negligible vs the 90 ms
   radix that follows. (The widen is unavoidable given the kernel ABI;
   the *pass count* is the choice.)
7. `_prepare` runs `np.asarray` + dtype + ndim checks per call — ~0.5 µs
   beyond the dtype-list.

---

## 7. Next-phase verdicts (one per hotspot)

Rule: GPU needs flops/byte ≳2 behind it. No kernel here exceeds 0.75.
**No GPU verdicts anywhere** — the strongest candidate
(`count_descents` int32 @0.75) is still 2.7× below threshold, and the
radix/lexsort kernels are indirect/scattered (GPU would need a
radix-architecture redesign, not a port). MOJO_NOTES §5's "CPU-only
below ~2 flops/byte" applies uniformly; host API availability is not
the argument.

| # | hotspot (§2) | verdict | why (flops/byte + evidence) |
| --- | --- | --- | --- |
| 1 | n=32 / small-n glue (80% glue, 0.14×) | **none** (for kernel) | Kernel is already 2.2 µs; the phase is Python: hoist dtype-list (save ~2.5 µs, ~16%), cut `_addr` calls (cache pointers / `c_void_p`, save ~5–8 µs), or raise the NumPy crossover. No kernel work moves 15.3 → 2.2 µs. Small-n total could reach ~4–6 µs. |
| 2 | narrow dtypes 0.07–0.23× | **none** (algorithmic width fix, not listed phases) | 0.17 flops/byte + traffic is 360 B/elem regardless of input width. Fix is a narrow-width radix (1–2 passes on 1–2 B keys) or a counting path — an algorithm change, not unroll/simd/parallel/gpu. Widening cost (0.5 ms) is noise. |
| 3 | radix-64/32 @1M+ (360/104 B, 0.17/0.31) | **parallel** | Memory-bound *and* far from peak (achieved 2–5 GB/s vs 27 GB/s streaming): histogram + scatter are the latency chain. `parallelize` from `max.algorithm` (MOJO_NOTES §4) over hist buckets / partitioned scatter is the standard radix parallelization; SIMD cannot fix data-dependent scatter. SIMD helps only the prologue/map (~3/60 ops). GPU rejected at 0.17 flops/byte. Unroll: minor (pass pair already unrolled). |
| 4 | lexsort chain | **parallel** (inherited) | Same 0.17 flops/byte per key; cost is k× radix + 14% gather overhead. Parallelizing the inner radix (row 3) plus fusing the gather into the scatter would cut both terms. No separate SIMD/GPU case. |
| 5 | count_descents (0.38–0.75) | **simd** | Pure streaming reduction, trivially vectorizable: map + compare + popcount-reduce. 10 GB/s scalar → expect 20+ GB/s with `simd_width_of` vector loads (MOJO_NOTES §3). Parallel also helps but SIMD is the first step (no synchronization needed). |
| 6 | radix-32 / counting (0.30–0.31) | **none** (already winning 3–4.6×) | Memory-bound and fastest-per-byte already; parallelizing counting scatters risks hist contention for little absolute gain (27 ms → maybe 15 ms). Lowest ROI. Leave for later. |
| 7 | insertion O(n²) | **none** | Algorithmic dead end above n≈256 (86 ms @20k). Correctly capped at 64. Only action: keep the cap, do not relax it. |
| 8 | allocation layout (5× separate mmaps) | **none** (or "unroll" by analogy: pack scratch) | Single 40 MB allocation with slices measured 15–25% faster @1M. Pure shim change (`_lib.py`), no kernel work. Complementary to every verdict above — do it first, it is measured. |

Ordering for the next phases on evidence:

1. **Shim (no kernel change):** hoist dtype-list, cache `_addr`/ptrs,
   single scratch allocation. Expected: n=32 15.3 → ~5 µs; n=1M
   ~15–25% faster for free. All three are measured above.
2. **SIMD:** `count_descents` vectorization (0.38–0.75 flops/byte,
   streaming, zero dependencies — the only clean SIMD target).
3. **PARALLEL:** radix 64/32 histogram/scatter (`max.algorithm.parallelize`
   behind a size gate, per MOJO_NOTES §4 — `std.algorithm` does not
   compile). Target the 360 B/elem traffic where achieved BW is 2–5 GB/s.
4. **No GPU** anywhere (max 0.75 flops/byte, spec threshold ~2).
5. **No unroll** anywhere (passes already paired; loops are
   memory-latency-bound, not branch-bound — the only measurable unroll
   effect was the (already done) 2-pass pairing in the launcher).

---

## 8. Repro

```bash
pixi run bench        # flock-wrapped, bench/bench.py (3 runs quoted in §1)
pixi run python bench/_analyze.py   # scaling + glue + dtype sweeps (§3–§5)
```

`bench/_analyze.py` is untracked scratch (task allows only
`bench/analysis.md`; delete it before committing, or keep it as the
method record — the numbers in this file do not depend on it existing).
No files under `src/` or `tests/` were modified.
