"""Index-sorting kernels: the numeric core behind an `argsort` library.

Everything exported here takes buffer addresses as plain `Int` and rebuilds the
pointer inside the body, because `@export` rejects parametric functions and an
inferred pointer origin would make the symbol parametric.

The main entry point is a least-significant-digit radix sort that emits
permutation indices. It is stable, so it agrees with `numpy.argsort(...,
kind="stable")` index for index, not merely value for value.

Key mapping. Two's-complement and IEEE-754 do not share a transform: signed
integers need only the sign bit flipped, floats need the whole word complemented
when the sign bit is set. They are separate `mode` values here. A descending sort
complements the mapped key, which reverses the value order and leaves equal keys
in input order, so descending parity against `numpy` is exact too.
"""

comptime HI = 256

comptime U32P = Pointer[UInt32, AnyOrigin[mut=True]]
comptime U64P = Pointer[UInt64, AnyOrigin[mut=True]]

def _u32p(a: Int) -> U32P:
    return U32P(unsafe_from_address=a)

def _u64p(a: Int) -> U64P:
    return U64P(unsafe_from_address=a)

def _map64(b: UInt64, mode: Int, descending: Int) -> UInt64:
    """Map a 64-bit word onto unsigned order. `mode`: 0 unsigned, 1 signed, 2 float.

    Two's complement needs only a flip of the sign bit: that is exactly the
    bijection between the signed and unsigned orderings of the same bits. IEEE
    754 needs the whole word complemented when the sign bit is set, because a
    negative float's magnitude runs backwards through its bit pattern. The two
    differ, so they are separate modes rather than one clever flag.
    """
    var bits = b
    if mode == 1:
        bits = bits ^ UInt64(0x8000000000000000)
    elif mode == 2:
        if (bits >> 63) == UInt64(1):
            bits = bits ^ UInt64(0xFFFFFFFFFFFFFFFF)
        else:
            bits = bits ^ UInt64(0x8000000000000000)
    if descending != 0:
        bits = ~bits
    return bits


def _map32(b: UInt32, mode: Int, descending: Int) -> UInt32:
    """32-bit sibling of `_map64`; see that docstring."""
    var bits = b
    if mode == 1:
        bits = bits ^ UInt32(0x80000000)
    elif mode == 2:
        if (bits >> 31) == UInt32(1):
            bits = bits ^ UInt32(0xFFFFFFFF)
        else:
            bits = bits ^ UInt32(0x80000000)
    if descending != 0:
        bits = ~bits
    return bits


def _pass64(
    src_k: U64P, src_v: U64P, dst_k: U64P, dst_v: U64P, h: U32P, n: Int,
    shift: Int
):
    for b in range(HI):
        h[unsafe_offset=b] = UInt32(0)
    for i in range(n):
        var d = Int((src_k[unsafe_offset=i] >> UInt64(shift)) & UInt64(255))
        h[unsafe_offset=d] += UInt32(1)
    var s = UInt32(0)
    for b in range(HI):
        var c = h[unsafe_offset=b]
        h[unsafe_offset=b] = s
        s += c
    for i in range(n):
        var d = Int((src_k[unsafe_offset=i] >> UInt64(shift)) & UInt64(255))
        var pos = h[unsafe_offset=d]
        h[unsafe_offset=d] = pos + UInt32(1)
        dst_k[unsafe_offset=Int(pos)] = src_k[unsafe_offset=i]
        dst_v[unsafe_offset=Int(pos)] = src_v[unsafe_offset=i]

def _pass32(
    src_k: U32P, src_v: U32P, dst_k: U32P, dst_v: U32P, h: U32P, n: Int,
    shift: Int
):
    for b in range(HI):
        h[unsafe_offset=b] = UInt32(0)
    for i in range(n):
        var d = Int((src_k[unsafe_offset=i] >> UInt32(shift)) & UInt32(255))
        h[unsafe_offset=d] += UInt32(1)
    var s = UInt32(0)
    for b in range(HI):
        var c = h[unsafe_offset=b]
        h[unsafe_offset=b] = s
        s += c
    for i in range(n):
        var d = Int((src_k[unsafe_offset=i] >> UInt32(shift)) & UInt32(255))
        var pos = h[unsafe_offset=d]
        h[unsafe_offset=d] = pos + UInt32(1)
        dst_k[unsafe_offset=Int(pos)] = src_k[unsafe_offset=i]
        dst_v[unsafe_offset=Int(pos)] = src_v[unsafe_offset=i]

@export("mas_radix_argsort")
def mas_radix_argsort(
    n: Int, keys_addr: Int, idx_addr: Int, ka_addr: Int, kb_addr: Int,
    va_addr: Int, vb_addr: Int, hist_addr: Int, word_bits: Int, mode: Int,
    descending: Int
) abi("C"):
    """Stable LSD radix sort; writes the permutation into `idx_addr`.

    `word_bits` is 32 or 64. The 32-bit path only touches the first `n * 4`
    bytes of each `uint64` scratch buffer, so one allocation serves both.
    """
    var idx = _u64p(idx_addr)
    var h = _u32p(hist_addr)
    if n <= 0:
        return
    if word_bits == 32:
        var src = _u32p(keys_addr)
        var ka = _u32p(ka_addr)
        var kb = _u32p(kb_addr)
        var va = _u32p(va_addr)
        var vb = _u32p(vb_addr)
        for i in range(n):
            ka[unsafe_offset=i] = _map32(src[unsafe_offset=i], mode, descending)
            va[unsafe_offset=i] = UInt32(i)
        for p in range(2):
            var base = p * 16
            _pass32(ka, va, kb, vb, h, n, base)
            _pass32(kb, vb, ka, va, h, n, base + 8)
        for i in range(n):
            idx[unsafe_offset=i] = UInt64(va[unsafe_offset=i])
    else:
        var src = _u64p(keys_addr)
        var ka = _u64p(ka_addr)
        var kb = _u64p(kb_addr)
        var va = _u64p(va_addr)
        var vb = _u64p(vb_addr)
        for i in range(n):
            ka[unsafe_offset=i] = _map64(src[unsafe_offset=i], mode, descending)
            va[unsafe_offset=i] = UInt64(i)
        for p in range(4):
            var base = p * 16
            _pass64(ka, va, kb, vb, h, n, base)
            _pass64(kb, vb, ka, va, h, n, base + 8)
        for i in range(n):
            idx[unsafe_offset=i] = va[unsafe_offset=i]

@export("mas_insertion_argsort")
def mas_insertion_argsort(
    n: Int, keys_addr: Int, idx_addr: Int, ka_addr: Int, va_addr: Int,
    mode: Int, descending: Int
) abi("C"):
    """Stable insertion sort for short keys; 64-bit key words only."""
    var idx = _u64p(idx_addr)
    if n <= 0:
        return
    var src = _u64p(keys_addr)
    var ka = _u64p(ka_addr)
    var va = _u64p(va_addr)
    for i in range(n):
        ka[unsafe_offset=i] = _map64(src[unsafe_offset=i], mode, descending)
        va[unsafe_offset=i] = UInt64(i)
    for i in range(1, n):
        var key = ka[unsafe_offset=i]
        var val = va[unsafe_offset=i]
        var j = i - 1
        while j >= 0:
            if ka[unsafe_offset=j] <= key:
                break
            ka[unsafe_offset=j + 1] = ka[unsafe_offset=j]
            va[unsafe_offset=j + 1] = va[unsafe_offset=j]
            j -= 1
        ka[unsafe_offset=j + 1] = key
        va[unsafe_offset=j + 1] = val
    for i in range(n):
        idx[unsafe_offset=i] = va[unsafe_offset=i]

@export("mas_counting_argsort")
def mas_counting_argsort(
    n: Int, keys_addr: Int, idx_addr: Int, ka_addr: Int, va_addr: Int,
    hist_addr: Int, buckets: Int, mode: Int, descending: Int
) abi("C"):
    """Counting sort on 32-bit keys by `key % buckets`, ascending or descending.

    `buckets` must be a power of two. `hist_addr` must hold `buckets` entries.
    Descending reverses the bucket order and leaves equal remainders in input
    order, so the result is a stable sort on `x % buckets`.
    """
    var idx = _u64p(idx_addr)
    var h = _u32p(hist_addr)
    if n <= 0 or buckets <= 0:
        return
    var src = _u32p(keys_addr)
    var ka = _u32p(ka_addr)
    var va = _u32p(va_addr)
    var mask = UInt32(buckets - 1)
    for b in range(buckets):
        h[unsafe_offset=b] = UInt32(0)
    for i in range(n):
        var d = Int(_map32(src[unsafe_offset=i], mode, 0) & mask)
        if descending != 0:
            d = buckets - 1 - d
        h[unsafe_offset=d] += UInt32(1)
    var s = UInt32(0)
    for b in range(buckets):
        var c = h[unsafe_offset=b]
        h[unsafe_offset=b] = s
        s += c
    for i in range(n):
        var d = Int(_map32(src[unsafe_offset=i], mode, 0) & mask)
        if descending != 0:
            d = buckets - 1 - d
        var pos = h[unsafe_offset=d]
        h[unsafe_offset=d] = pos + UInt32(1)
        ka[unsafe_offset=Int(pos)] = UInt32(i)
        va[unsafe_offset=Int(pos)] = UInt32(i)
    for i in range(n):
        idx[unsafe_offset=i] = UInt64(ka[unsafe_offset=i])

@export("mas_count_descents")
def mas_count_descents(
    n: Int, keys_addr: Int, word_bits: Int, mode: Int, descending: Int
) abi("C") -> Int:
    """Count adjacent pairs whose mapped key strictly decreases.

    This is the natural-run statistic a merge or MSD radix sort uses to decide
    whether an input is already ordered; `numpy` has no direct equivalent, so
    the reference here is a plain scan.
    """
    if n <= 1:
        return 0
    if word_bits == 32:
        var src = _u32p(keys_addr)
        var prev = _map32(src[unsafe_offset=0], mode, descending)
        var count = 0
        for i in range(1, n):
            var cur = _map32(src[unsafe_offset=i], mode, descending)
            if cur < prev:
                count += 1
            prev = cur
        return count
    else:
        var src = _u64p(keys_addr)
        var prev = _map64(src[unsafe_offset=0], mode, descending)
        var count = 0
        for i in range(1, n):
            var cur = _map64(src[unsafe_offset=i], mode, descending)
            if cur < prev:
                count += 1
            prev = cur
        return count
