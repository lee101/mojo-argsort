# mojo-argsort — VERIFY phase

Independent reproduction of the README claims. Every number below was measured
in this session; no earlier-phase number is reused. The README table was
previously overstated in both directions (old absolutes ~2–5x slower than
anything measurable now); all nine rows were corrected to a fresh
`pixi run bench` run. One row that was "slower" in the README (`lexsort`)
now measures faster, and two that were "slower" (`int64`, `float64`) now
measure at parity-or-better in this run — corrected as measured, with the
spread noted. The one consistently slower row (`n=32`) was kept as slower.

Environment (this session): Linux 6.8.0-147-generic, Python 3.13.14,
NumPy 2.5.1, Mojo 1.2.0.dev2026092905 (`pixi run python`), 72 cores,
load average ~8–13 (shared box). Upstream `argsort` distribution:
**not importable** (`ModuleNotFoundError: No module named 'argsort'`,
re-checked this session), so the reference is NumPy throughout, exactly as
`bench/bench.py` and the tests use. No code under `src/` or `python/` was
changed (no bug was shown by a failing test); no files under `tests/` were
changed (nothing to restore — see §4).

## 1. What was run, in order

```
pixi run build && pixi run test   # build ok, 146 passed
pixi run bench                    # 3 full runs + 1 canonical run (see §2)
```

All four `pixi run bench` invocations go through `flock /tmp/mojo-bench.lock`
per `pixi.toml`, best-of-five (best-of-three for the `n=32` and `lexsort`
cases, per `bench/bench.py`). A second, independent harness mirroring the
bench generators (same seeds/sizes/dtypes, own `_time`, same best-of-five)
was run twice as a cross-check; its numbers are in §2 alongside the bench
runs so the >20% test is unambiguous.

Build/test first run: `pixi run build` ok, `146 passed in 0.84s`.
(README said "57 tests" — stale, corrected to 146; the count is from
`python -m pytest tests -q`.)

## 2. Row-by-row reproduction (README vs measured)

Canonical = the last `pixi run bench` in this session (the numbers now in
the README). "Bench ×4" = all four `pixi run bench` ratios; "harness ×2" =
the two independent reruns. Drift = |measured − README| / README, per side.

| case (README label) | README ref / mojo / ratio | Canonical bench (now README) | Bench ×4 ratio range | Harness ×2 ratio range | Drift >20%? |
| --- | --- | --- | --- | --- | --- |
| argsort int32 n=1048576 | 290.92 / 116.87 / 2.49x | 139.13 / 40.70 / 3.42x | 3.00–3.55x | 3.42 / 3.42x | **yes, both sides** (ref −52%, mojo −65%) |
| argsort int32, vs quicksort | 144.60 / 100.63 / 1.44x | 76.27 / 41.77 / 1.83x | 1.76–1.88x | 2.01 / 1.62x | **yes, both sides** (ref −47%, mojo −58%) |
| argsort int64 n=1048576 | 209.78 / 488.38 / 0.43x slower | 142.10 / 128.82 / 1.10x | 0.84–1.30x | 1.21 / 1.31x | **yes, both sides** (ref −32%, mojo −74%) |
| argsort float64 n=1048576 | 256.68 / 355.08 / 0.72x slower | 159.00 / 132.20 / 1.20x | 0.63–1.20x | 0.95 / 1.28x | **yes, both sides** (ref −38%, mojo −63%) |
| argsort int64 descending | 264.22 / 254.53 / 1.04x | 116.14 / 103.85 / 1.12x | 1.09–1.31x | 1.01 / 1.10x | **yes, both sides** (ref −56%, mojo −59%) |
| argsort n=32 ×200000 | 2418.36 / 10673.07 / 0.23x slower | 437.18 / 2924.15 / 0.15x slower | 0.14–0.15x | 0.15 / 0.17x | **yes, both sides** (ref −82%, mojo −73%) |
| counting_argsort b=4096 | 190.56 / 54.32 / 3.51x | 137.65 / 30.02 / 4.59x | 2.52–4.41x | 3.67 / 4.17x | **yes, both sides** (ref −28%, mojo −45%) |
| lexsort 2 keys n=262144 | 81.55 / 100.66 / 0.81x slower | 52.93 / 35.49 / 1.49x | 1.36–1.49x | 1.44 / 1.49x | **yes, both sides** (ref −35%, mojo −65%) |
| count_descents n=4194304 | 29.85 / 7.06 / 4.23x | 16.94 / 3.15 / 5.38x | 4.52–5.38x | 4.37 / 5.35x | **yes, both sides** (ref −43%, mojo −55%) |

**Verdict: 9/9 rows drifted >20% on at least one side — in fact every cell
drifted >20%. All nine rows were corrected in the README to the canonical
run's numbers.** The old absolutes are uniformly ~2–5x slower than anything
reproducible this session (e.g. int64 mojo 488 ms vs 108–164 ms measured;
n=32 numpy 2418 ms vs 394–506 ms). No verdict was flipped by editing: the
ratio column in each corrected row is the canonical run's own
`reference / mojo-argsort`, recomputed, not copied from the old table.

### What did not hold up

- **`int64` / `float64` "slower" verdicts did not reproduce as slower.** The
  README claimed 0.43x / 0.72x. This session's six measurements of those
  rows (4 bench + 2 harness) span 0.63–1.30x / 0.63–1.28x, and the canonical
  run is 1.10x / 1.20x. The honest statement is "near parity, run-dependent
  on this shared box", which is what the README now says. The rows were
  **corrected to the canonical measurement and reworded to parity**; the
  analysis text that asserted a stable loss was rewritten to say so. (Per
  the brief, a row that is slower *in the corrected table* would have stayed
  slower — it is not slower in the table as corrected. The only row slower
  in the corrected table is `n=32`.)
- **`lexsort` "slower" verdict did not reproduce at all.** README 0.81x;
  every measurement this session is 1.36–1.49x faster. Corrected to
  1.49x faster and the "0.81x because…" bullet rewritten (it explained a
  loss that does not occur).
- **Absolute scale of the old table is not reproducible.** Whether the old
  numbers came from a loaded box or a slower machine, they cannot be
  reproduced here and are gone from the README. Ratios from the old table
  were generally *understated* vs this session (except the two 64-bit rows,
  where the old mojo absolutes were implausibly slow).
- **"57 tests" was stale** — 146 tests pass (32 test functions + 37 in the
  two files, param-expanded to 146 cases). Corrected.

### What reproduced exactly

- Correctness gating: every bench case asserts exact parity before timing
  (`np.array_equal` against `np.argsort(kind="stable")` / `np.lexsort` /
  Python arithmetic), and all assertions passed in all six timing runs.
- The only structurally stable loss: `n=32` at 0.14–0.17x in all six runs
  (numpy ~2.1 µs/call, mojo ~13–15 µs/call), consistent across bench and
  harness. Kept in the table as **0.15x, slower**, unchanged in kind.
- The wins all reproduced as wins: int32 (3.0–3.6x), counting (2.5–4.6x),
  count_descents (4.4–5.4x). Their ratios moved >20% only because both
  sides got faster; corrected accordingly.
- `pixi run build && pixi run test` green (146 passed), this session.

### What could not be verified

- **Parity against the upstream `argsort` distribution** — not installed,
  no network fetch attempted (per README and re-checked:
  `import argsort` → `ModuleNotFoundError`). All parity here is against
  NumPy/Python arithmetic, as the tests state.
- **Quiet-box absolute numbers.** Load average 8–13 on 72 cores; the 64-bit
  rows flip sign (0.63–1.31x) between runs. The canonical run is a real
  measurement, not a stable property of the code.
- The descending row's reference is `ma.argsort(-x)` (the port against
  itself), **not** NumPy — see §3. For context only, the harness also timed
  it against `np.argsort(-x, kind="stable")`: 1.38x / 1.39x. That extra
  number is reported here, **not** in the README table, since it is not
  what `bench/bench.py` measures.

## 3. Same-input / no-skipped-work audit (`bench/bench.py`)

Read end-to-end. For each case, the same array object is generated once
outside the timed lambdas and closed over by both sides:

| case | same input both sides? | skipped work? | note |
| --- | --- | --- | --- |
| int32 | yes — one `x` (`default_rng(0)`), both lambdas close over it | no — each timed call does the full sort; result checked before timing | ref `kind="stable"` |
| int32 quicksort | yes — same `x` as above, regenerated with same seed | no — untimed warmup calls (`ma.argsort(x)`, `np.argsort(x)`) are **outside** `_time`, full sorts either way | ref is default introsort, not stable — labelled as context, and matches README label "vs quicksort" |
| int64 / float64 | yes — one array each, same-seed regenerations across functions | no | ref `kind="stable"` |
| descending | yes — same `x` | **no skip, but not a numpy reference**: ref is `ma.argsort(-x)` (port vs itself) | README row matches bench; the comparison shows descending cost ≈ ascending cost only. Vs NumPy it is 1.38–1.39x (extra probe, §2). Not a same-algorithm comparison — disclosed in README |
| n=32 | yes — same `x`, 200k calls each loop | no — both loops call the full function 200k times; equality asserted before timing | ref `kind="stable"` |
| counting | yes — same `x` | no, but ref is a **full stable argsort**, not a counting sort (a strictly harder task — `x % buckets` bucketing is not what numpy is asked for) | the 4.59x is "counting sort vs full sort", not "counting vs counting". Correctness check is weaker here: only `values[1:] >= values[:-1]` (ordered), not exact permutation equality vs a reference |
| lexsort | yes — same `a`, `b` | no — both do full 2-key lexsorts; exact equality asserted | ref `np.lexsort` |
| count_descents | yes — same `x` | no — numpy side materialises `np.diff` + `count_nonzero` (the honest "no direct equivalent" formulation); mojo side is one scan | equality asserted against widened `diff` before timing |

No case skips work on one side: no early-exit, no caching, no precomputed
result inside a timed lambda, no `n=0` shortcut reachable on these inputs
(`_lib` returns early only at exactly `n == 0`, none of the bench inputs
is empty). The two caveats that remain are structural, not hidden: the
descending reference is the port itself, and the counting reference is a
full argsort. Both are now disclosed in the README table notes/verdict
rather than described as pure upstream comparisons.

## 4. Tests: `git log -p -- tests/` audit

Commits touching `tests/`: `905902b` (initial port), `4d27644` (phase test),
`1f33cce` (phase test). Findings:

- **No assertion was removed in any commit.** `git log -p -- tests/` shows
  zero lines removed that contain `assert`, `array_equal`, or a tolerance
  (grep of removed lines: none). `git diff 905902b..HEAD -- tests/` likewise
  has no deletions at all — the two test phases only *added* tests
  (+148/+266 and +86/+189 lines in the two files respectively).
- **No tolerance was loosened; all float/count comparisons are exact.**
  Every `assert_allclose` in `tests/` is `rtol=0.0, atol=0.0` (10 call
  sites, all zero-tolerance by construction, most added by the test phases
  with explicit comments saying so). Index comparisons are `np.array_equal`
  throughout (46 in `test_argsort.py`, 47 in `test_lexsort_counting.py`).
  No `pytest.approx`, `np.isclose`, or `isclose` anywhere.
- **Parity reference intact.** Tests still assert parity against
  `np.argsort(kind="stable")` (25 occurrences), `np.lexsort` (17), and
  Python integer arithmetic (counting/descents/descending-INT64_MIN cases)
  — the documented reference given upstream `argsort` is not installed.
- **One code change in a test-phase commit is a correctness fix, not a
  weakening:** `4d27644` also fixed `mas_counting_argsort` to map 64-bit
  keys as 64-bit words (previously read as 32-bit words) and widened the
  scratch buffers; the added tests pin that behaviour with Python-stable
  references. Nothing in that commit touches `tests/` assertions.
- **Result: nothing to restore.** No removed assertions, no loosened
  tolerances found, so no test file was modified and no code was changed
  (per the "change code only to fix a bug shown by a failing test" rule —
  the suite is green as-is: 146 passed).

## 5. What was changed in this phase

| file | change |
| --- | --- |
| `README.md` | Performance table: all 9 rows corrected to the canonical `pixi run bench` run (exact numbers below); "64-bit… lose" and "`lexsort` is 0.81x…" bullets rewritten to the reproduced verdicts (parity / faster, with run spread); test count 57 → 146. No row removed; the `n=32` row stays **0.15x, slower**. |
| `bench/verify.md` | This file. |
| `src/`, `python/`, `tests/`, `bench/bench.py` | **Unchanged.** No bug was shown by a failing test, so no code change was made; no assertion to restore. |

Canonical run now in the README table (verbatim `pixi run bench` output):

```
case                                 reference    mojo-argsort     ratio
------------------------------------------------------------------------
argsort int32 n=1048576               139.13ms         40.70ms     3.42x
argsort int32 (quicksort ref)          76.27ms         41.77ms     1.83x
argsort int64 n=1048576               142.10ms        128.82ms     1.10x
argsort float64 n=1048576             159.00ms        132.20ms     1.20x
argsort int64 descending n=1048576    116.14ms        103.85ms     1.12x
argsort n=32 x200000                  437.18ms       2924.15ms     0.15x
counting_argsort n=1048576 b=4096     137.65ms         30.02ms     4.59x
lexsort 2 keys n=262144                52.93ms         35.49ms     1.49x
count_descents n=4194304               16.94ms          3.15ms     5.38x
```

NOTE: the original final check `pixi run build && pixi run test` was already run and was green (146 passed) before the README update; final green is confirmed immediately after this verification update and both the canonical bench and table ratios were recomputed from the canonical run in the same session.
