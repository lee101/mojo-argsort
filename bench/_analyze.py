"""Scratch harness for ANALYZE phase. Times each public entry point, measures
glue vs kernel split, checks for per-element ctypes loops, and derives bytes /
flops / flops-per-byte from the actual kernel code (src/kernels.mojo)."""
import os, sys, time, pathlib, json, ctypes
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import numpy as np
import mojo_argsort as ma
from mojo_argsort import _lib

def best(fn, n=5):
    for _ in range(2): fn()
    vals=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(); vals.append(time.perf_counter()-t0)
    return min(vals), vals

def report(label, t, extra=""):
    print(f"{label:60s} {t*1e3:10.3f}ms {extra}")

# 0. env
print(f"python {sys.version.split()[0]} numpy {np.__version__}")
try:
    import argsort
    print("UPSTREAM argsort IMPORTABLE", argsort)
except Exception as e:
    print(f"UPSTREAM argsort NOT IMPORTABLE: {type(e).__name__}: {e}")

# 1. Check python-level loops need to grep for per-element ctypes
import inspect
for mod in (ma, _lib):
    for name in ["argsort","lexsort","counting_argsort","count_descents","is_sorted","radix_argsort","insertion_argsort","count_descents"]:
        if hasattr(mod, name):
            try:
                src = inspect.getsource(getattr(mod, name))
                print(f"\n--- {mod.__name__}.{name} ---\n{src[:2000]}")
            except Exception as e:
                print(name, e)
print("\n--- core.py lexsort loop ---")
import pathlib as pl
print(pl.Path("python/mojo_argsort/core.py").read_text())
print("\n--- _lib.py ---")
print(pl.Path("python/mojo_argsort/_lib.py").read_text())

# 2. Per-size scaling: argsort int64/int32/float64 at n = 32..4M
for dtype_name, gen in [
    ("int32", lambda n,rng: rng.integers(0,1<<20,size=n).astype(np.int32)),
    ("int64", lambda n,rng: rng.integers(-(1<<40),1<<40,size=n).astype(np.int64)),
    ("float64", lambda n,rng: rng.standard_normal(n)),
]:
    print(f"\n=== argsort {dtype_name} scaling ===")
    for n in [32,64,65,128,256,1024,4096,16384,65536,262144,1048576,4194304]:
        rng=np.random.default_rng(0)
        x=gen(n,rng)
        # correctness
        got=ma.argsort(x)
        assert np.array_equal(got, np.argsort(x, kind="stable")), f"mismatch {dtype_name} {n}"
        t,_=best(lambda: ma.argsort(x), n=5 if n<100000 else 5)
        tn,_=best(lambda: np.argsort(x, kind="stable"), n=5 if n<100000 else 3)
        # small-n per-call
        if n<=4096:
            reps=20000
            t0=time.perf_counter()
            for _ in range(reps): ma.argsort(x)
            t_one=(time.perf_counter()-t0)/reps
            t0=time.perf_counter()
            for _ in range(reps): np.argsort(x, kind="stable")
            tn_one=(time.perf_counter()-t0)/reps
            print(f"n={n:>8} mojo={t_one*1e6:8.2f}us/call numpy={tn_one*1e6:8.2f}us/call ratio={tn_one/t_one:.2f}x")
        else:
            print(f"n={n:>8} mojo={t*1e3:8.2f}ms numpy={tn*1e3:8.2f}ms ratio={tn/t:.2f}x")

# 3. Glue vs kernel split: preload buffers, call lib directly bypassing _lib allocations
print("\n=== GLUE vs KERNEL split for radix ===")
rng=np.random.default_rng(0)
n=1<<20
x=rng.integers(-(1<<40),1<<40,size=n).astype(np.int64)
# full call
t_full,_=best(lambda: ma.argsort(x))
print(f"full ma.argsort int64 n=1M: {t_full*1e3:.2f}ms")
t_full,_=best(lambda: _lib.radix_argsort(x))
print(f"direct _lib.radix_argsort: {t_full*1e3:.2f}ms")
# now preallocate and call lib.mas_* directly, timed many times
keys=np.ascontiguousarray(x)
mode=_lib._mode_of(keys.dtype)
idx=np.empty(n,dtype=np.int64); ka=np.empty(n,dtype=np.uint64); kb=np.empty(n,dtype=np.uint64); va=np.empty(n,dtype=np.uint64); vb=np.empty(n,dtype=np.uint64); hist=np.empty(256,dtype=np.uint32)
def _addr(a): return int(a.ctypes.data)
args=(n,_addr(keys),_addr(idx),_addr(ka),_addr(kb),_addr(va),_addr(vb),_addr(hist),64,mode,0)
_LIB=_lib.lib
t_kernel,_=best(lambda: _LIB.mas_radix_argsort(*args))
print(f"kernel-only (lib call, prealloc buffers): {t_kernel*1e3:.2f}ms  glue={ (t_full-t_kernel)*1e3:.2f}ms  glue_frac={(t_full-t_kernel)/t_full*100:.1f}%")
# allocation-only cost (no kernel)
def alloc_only():
    idx=np.empty(n,dtype=np.int64); ka=np.empty(n,dtype=np.uint64); kb=np.empty(n,dtype=np.uint64); va=np.empty(n,dtype=np.uint64); vb=np.empty(n,dtype=np.uint64); return idx,ka,kb,va,vb
t_alloc,_=best(alloc_only)
print(f"allocation-only (5x n=1M uint64/int64 empty): {t_alloc*1e3:.2f}ms")
# also measure np.empty cost per array
t0=time.perf_counter()
for _ in range(1000): np.empty(n,dtype=np.uint64)
print(f"one np.empty(n=1M uint64): {(time.perf_counter()-t0)/1000*1e3:.3f}ms")
# copy/glue for int32 path
x32=rng.integers(0,1<<20,size=n).astype(np.int32)
t_full32,_=best(lambda: ma.argsort(x32))
print(f"\nfull ma.argsort int32 n=1M: {t_full32*1e3:.2f}ms")
keys32=np.ascontiguousarray(x32); idx32=np.empty(n,dtype=np.int64); ka32=np.empty(n,dtype=np.uint64); kb32=np.empty(n,dtype=np.uint64); va32=np.empty(n,dtype=np.uint64); vb32=np.empty(n,dtype=np.uint64); hist32=np.empty(256,dtype=np.uint32)
args32=(n,_addr(keys32),_addr(idx32),_addr(ka32),_addr(kb32),_addr(va32),_addr(vb32),_addr(hist32),32,1,0)
t_kernel32,_=best(lambda: _LIB.mas_radix_argsort(*args32))
print(f"kernel-only int32: {t_kernel32*1e3:.2f}ms glue={(t_full32-t_kernel32)*1e3:.2f}ms")

# small n glue breakdown
print("\n=== SMALL n breakdown n=32 (insertion path) ===")
x32s=rng.integers(0,100,size=32)
# time full call
reps=200000
t0=time.perf_counter()
for _ in range(reps): ma.argsort(x32s)
print(f"ma.argsort n=32 per call: {(time.perf_counter()-t0)/reps*1e6:.2f}us")
t0=time.perf_counter()
for _ in range(reps): np.argsort(x32s,kind="stable")
print(f"np.argsort n=32 per call: {(time.perf_counter()-t0)/reps*1e6:.2f}us")
# kernel-only insertion
keys_s=np.ascontiguousarray(x32s).astype(np.int64)  # _widen(int64) -> x is int64 already? integers default int64
idx_s=np.empty(32,dtype=np.int64); ka_s=np.empty(32,dtype=np.uint64); va_s=np.empty(32,dtype=np.uint64)
args_s=(32,_addr(keys_s),_addr(idx_s),_addr(ka_s),_addr(va_s),1,0)
_LIB.mas_insertion_argsort(*args_s)
t0=time.perf_counter()
for _ in range(reps): _LIB.mas_insertion_argsort(*args_s)
print(f"kernel-only insertion n=32 per call: {(time.perf_counter()-t0)/reps*1e6:.3f}us")
# alloc cost small
t0=time.perf_counter()
for _ in range(reps):
    np.empty(32,dtype=np.int64); np.empty(32,dtype=np.uint64); np.empty(32,dtype=np.uint64)
print(f"alloc 3x n=32 per call: {(time.perf_counter()-t0)/reps*1e6:.3f}us")
# _prepare cost
t0=time.perf_counter()
for _ in range(reps): ma.argsort(x32s, method="insertion")
print(f"ma.argsort insertion forced per call: {(time.perf_counter()-t0)/reps*1e6:.2f}us")
# how many ctypes calls per argsort? inspect: one call for 1-key
print("ctypes calls per argsort: 1 (lib.mas_*), lexsort: 1 per key (from core.py for loop)")

# 4. counting, lexsort, descents detailed
print("\n=== counting_argsort ===")
for n,b in [(1<<20,4096),(1<<20,256),(1<<16,4096)]:
    rng=np.random.default_rng(5)
    x=rng.integers(0,1<<24,size=n).astype(np.int32)
    got=ma.counting_argsort(x,b)
    t,_=best(lambda: ma.counting_argsort(x,b))
    tn,_=best(lambda: np.argsort(x,kind="stable"))
    print(f"n={n} b={b} mojo={t*1e3:.2f}ms numpy={tn*1e3:.2f}ms ratio={tn/t:.2f}x")
    # kernel only
print("\n=== lexsort ===")
for n,keys_ct in [(1<<18,2),(1<<18,3),(1<<20,2)]:
    rng=np.random.default_rng(6)
    ks=[rng.integers(0,1<<16,size=n), rng.integers(0,1<<8,size=n)][:keys_ct]
    if keys_ct==3: ks=ks+[rng.integers(0,1<<8,size=n)]
    got=ma.lexsort(*ks)
    assert np.array_equal(got, np.lexsort(tuple(ks))), "lex mismatch"
    t,_=best(lambda: ma.lexsort(*ks), n=3)
    tn,_=best(lambda: np.lexsort(tuple(ks)), n=3)
    print(f"n={n} k={keys_ct} mojo={t*1e3:.2f}ms numpy={tn*1e3:.2f}ms ratio={tn/t:.2f}x")
    # per-key radix cost (indirect via argsort int64 n)
    rng2=np.random.default_rng(0)
    x1=rng2.integers(-(1<<40),1<<40,size=n).astype(np.int64)
    t1,_=best(lambda: ma.argsort(x1), n=3)
    print(f"  single radix int64 n={n}: {t1*1e3:.2f}ms -> lexsort k={keys_ct} expect ~{keys_ct}*t1={keys_ct*t1*1e3:.2f}ms")
    # check intermediate copies: core does idx[_radix_argsort(k[idx], ...)] each iter: that's gather + allocate
print("\n=== count_descents / is_sorted ===")
for n in [1<<22, 1<<20, 1024]:
    rng=np.random.default_rng(7)
    x=np.ascontiguousarray(rng.integers(0,1<<20,size=n))
    t,_=best(lambda: ma.count_descents(x))
    tn,_=best(lambda: np.count_nonzero(np.diff(x)<0))
    print(f"n={n} mojo={t*1e3:.3f}ms numpy={tn*1e3:.3f}ms ratio={tn/t:.2f}x")
    # kernel-only
    max_=np.ascontiguousarray(x); mode_= _lib._mode_of(max_.dtype)
    if max_.dtype.itemsize not in (4,8): max_=_lib._widen(max_)
    args_d=(n,_addr(max_),64,1,0)
    td,_=best(lambda: _LIB.mas_count_descents(*args_d))
    print(f"  kernel-only: {td*1e3:.3f}ms glue={(t-td)*1e3:.3f}ms")
    t_is,_=best(lambda: ma.is_sorted(x))
    print(f"  is_sorted: {t_is*1e3:.3f}ms")
print("\n=== insertion path large ===")
for n in [64, 129, 1000, 5000, 20000]:
    rng=np.random.default_rng(0)
    x=rng.integers(0,1<<20,size=n).astype(np.int64)
    got=ma.argsort(x, method="insertion")
    assert np.array_equal(got, np.argsort(x,kind="stable"))
    t,_=best(lambda: ma.argsort(x, method="insertion"), n=5)
    tn,_=best(lambda: np.argsort(x,kind="stable"), n=5)
    print(f"n={n} insertion mojo={t*1e3:.3f}ms numpy={tn*1e3:.3f}ms ratio={tn/t:.2f}x")
print("\n=== dtype widening costs ===")
for dt in [np.int8,np.int16,np.uint8,np.uint16,np.float32,np.float16]:
    rng=np.random.default_rng(0)
    n=1<<20
    if np.dtype(dt).kind=="f": x=rng.standard_normal(n).astype(dt)
    else: x=rng.integers(-100,100,size=n).astype(dt)
    got=ma.argsort(x); assert np.array_equal(got, np.argsort(x,kind="stable"))
    t,_=best(lambda: ma.argsort(x))
    tn,_=best(lambda: np.argsort(x,kind="stable"))
    tw,_=best(lambda: _lib._widen(x))
    print(f"{np.dtype(dt).name:>8} n=1M mojo={t*1e3:.2f}ms numpy={tn*1e3:.2f}ms ratio={tn/t:.2f}x widen={tw*1e3:.2f}ms")
print("\n=== descending ===")
for dt,gen in [("int64", lambda n,rng: rng.integers(-(1<<40),1<<40,size=n).astype(np.int64)), ("int32", lambda n,rng: rng.integers(0,1<<20,size=n).astype(np.int32))]:
    rng=np.random.default_rng(3); n=1<<20; x=gen(n,rng)
    t,_=best(lambda: ma.argsort(x,descending=True))
    tn,_=best(lambda: np.argsort(-x,kind="stable"))
    print(f"{dt} descending mojo={t*1e3:.2f}ms numpy_neg={tn*1e3:.2f}ms ratio={tn/t:.2f}x NOTE bench ref is ma.argsort(-x) itself")
    assert np.array_equal(ma.argsort(x,descending=True), ma.argsort(-x))
print("\nDONE probing")
