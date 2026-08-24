#: Axis reductions and scans: the folds, the arg-folds, the cumulative passes,
#: and the two whole-array predicates that are reductions rather than ufuncs.
#:
#: The kernel shape is the point of this module. A reduction has two extents
#: that pull in opposite directions — how many answers it produces
#: (`segments`) and how many elements each answer folds (`n`) — and a grid
#: sized off the wrong one starves. Giving every output a thread is what this
#: repo has been bitten by: reducing `[1000000, 8]` down axis 0 leaves eight
#: threads holding eight million elements. So the general path permutes the
#: reduced axes to the fast side, hands each segment a *slice of a
#: threadgroup*, and picks that slice's width from `n`:
#:
#: | `n` | width | one group covers | why |
#: | --- | --- | --- | --- |
#: | 3 | 4 | 64 segments | a short axis must not cost 252 idle lanes |
#: | 1000 | 256 | 1 segment | the group strides the axis, as `group_total` does |
#: | 10^7, one segment | 256 | 1 chunk of 1 | a second pass joins the chunks |
#:
#: which is one kernel: width 256 is jaitensor's `group_total` pattern, width 1
#: is a thread per output, and both are right in their own regime. The
#: fallback for a single huge segment is the two-stage partial/finish pass,
#: capped at 1024 groups so the finish stays one dispatch.
#:
#: `axis` is `null` for everything, an int, or a list or tuple of them.
#: `keepdims` leaves the reduced axes as length one, and `dtype` names the
#: accumulator, which is also the result — numpy's rule, and the reason
#: `sum` of an integer array comes back `INT64`.
#:
#: The nan-aware family is a flag on the same kernels rather than a second set
#: of them. It is not a substitution of the identity: `nanmin`'s identity is
#: NaN and its combine drops a NaN operand, which is what makes an all-NaN
#: slice come back NaN the way numpy's does.
#:
#: Axis permutation is done here, on `shape` and `strides` directly, rather
#: than through `manipulate`: a reduction needs the two halves as separate
#: extent lists for the kernel to walk, which is not what a transpose returns,
#: and taking the dependency would buy nothing.
import std.gpu as gpu
from std.gpu import Buffer
from .array import BOOL, COMPLEX64, FLOAT32, FLOAT64, INT32, INT64
from .array import GROUP, NDArray, PRELUDE, binary_arith, diagonal_view
from .array import dtype_is_integer, dtype_lanes, element_count, f32_bits, fill
from .array import is_contiguous, kernel_of, layout_buffer, normalize_axes
from .array import normalize_axis, promote, require_real

#: Fold selector: running total. Also the finish of every counting fold.
const OP_SUM = 0
#: Fold selector: running product.
const OP_PROD = 1
#: Fold selector: smallest value.
const OP_MIN = 2
#: Fold selector: largest value.
const OP_MAX = 3
#: Fold selector: 1 when any element is non-zero.
const OP_ANY = 4
#: Fold selector: 1 when every element is non-zero.
const OP_ALL = 5
#: Fold selector: how many elements are non-zero.
const OP_NONZERO = 6
#: Fold selector: how many elements are not NaN, which is the divisor every
#: nan-aware average needs.
const OP_FINITE = 7
#: Fold selector: total of `(x - center[segment])^2`, the second pass of a
#: variance. Two passes and not `E[x^2] - E[x]^2`: the one-pass formula
#: cancels catastrophically once the mean is large next to the spread, and a
#: variance that reports a negative number is worse than one extra read.
const OP_CENTERED = 8

let SOURCE = PRELUDE + """
constexpr constant uint NR_SUM = 0u;
constexpr constant uint NR_PROD = 1u;
constexpr constant uint NR_MIN = 2u;
constexpr constant uint NR_MAX = 3u;
constexpr constant uint NR_ANY = 4u;
constexpr constant uint NR_ALL = 5u;
constexpr constant uint NR_NONZERO = 6u;
constexpr constant uint NR_FINITE = 7u;
constexpr constant uint NR_CENTERED = 8u;

/* Spelled from the bits rather than from `INFINITY` and `NAN`, which are
 * library macros this repo has not leaned on anywhere else. The largest finite
 * float is not a substitute: an array that genuinely holds an infinity would
 * beat it and a minimum would come back wrong. */
inline float nr_inf() { return as_type<float>(0x7f800000u); }
inline float nr_nan() { return as_type<float>(0x7fc00000u); }

/* What a fold starts from, and therefore what an element the nan flag drops
 * turns into.
 *
 * `nanmin` starts from NaN, not from +inf. Paired with the combine below --
 * which prefers a real operand over a NaN one -- that makes an all-NaN slice
 * fold to NaN, matching numpy, without the extra pass a count would cost. */
inline float nr_identity(uint op, uint skip_nan) {
    if (op == NR_PROD || op == NR_ALL) return 1.0f;
    if (op == NR_MIN) return skip_nan != 0u ? nr_nan() : nr_inf();
    if (op == NR_MAX) return skip_nan != 0u ? nr_nan() : -nr_inf();
    return 0.0f;
}

/* Join two accumulators. Associative in every case, which is what lets the
 * tree below reorder the work.
 *
 * `min` and `max` need the NaN branches written out: Metal's return the
 * non-NaN operand, and numpy's propagate. */
inline float nr_combine(float a, float b, uint op, uint skip_nan) {
    if (op == NR_PROD) return a * b;
    if (op == NR_MIN || op == NR_ALL) {
        if (skip_nan != 0u) return (a != a) ? b : ((b != b) ? a : min(a, b));
        return (a != a) ? a : ((b != b) ? b : min(a, b));
    }
    if (op == NR_MAX || op == NR_ANY) {
        if (skip_nan != 0u) return (a != a) ? b : ((b != b) ? a : max(a, b));
        return (a != a) ? a : ((b != b) ? b : max(a, b));
    }
    return a + b;
}

/* One element turned into the accumulator the fold carries. */
inline float nr_ingest(float value, float center, uint op, uint skip_nan) {
    if (skip_nan != 0u && value != value) return nr_identity(op, skip_nan);
    if (op == NR_ANY || op == NR_ALL || op == NR_NONZERO) return value != 0.0f ? 1.0f : 0.0f;
    if (op == NR_FINITE) return value == value ? 1.0f : 0.0f;
    if (op == NR_CENTERED) { float d = value - center; return d * d; }
    return value;
}

inline float4 nr_combine4(float4 a, float4 b, uint op, uint skip_nan) {
    if (op == NR_PROD) return a * b;
    if (op == NR_MIN || op == NR_ALL) {
        if (skip_nan != 0u) return select(select(min(a, b), a, b != b), b, a != a);
        return select(select(min(a, b), b, b != b), a, a != a);
    }
    if (op == NR_MAX || op == NR_ANY) {
        if (skip_nan != 0u) return select(select(max(a, b), a, b != b), b, a != a);
        return select(select(max(a, b), b, b != b), a, a != a);
    }
    return a + b;
}

inline float4 nr_ingest4(float4 v, float center, uint op, uint skip_nan) {
    float4 value = v;
    if (skip_nan != 0u) value = select(value, float4(nr_identity(op, skip_nan)), value != value);
    if (op == NR_ANY || op == NR_ALL || op == NR_NONZERO)
        return select(float4(0.0f), float4(1.0f), value != 0.0f);
    if (op == NR_FINITE) return select(float4(0.0f), float4(1.0f), value == value);
    if (op == NR_CENTERED) { float4 d = value - center; return d * d; }
    return value;
}

/* A linear index over `rank` axes turned into a float-slot displacement.
 *
 * One division per axis and a subtraction for the remainder, never a `/`
 * beside a `%`: the device has no integer divide and spelling it twice pays
 * for it twice. Extents and strides sit at separate bases so one walk serves
 * two operands over the same shape, which is what the scan needs. */
inline int nr_walk(
    device const float* meta,
    uint ext_at,
    uint str_at,
    uint rank,
    uint linear
) {
    int at = 0;
    uint rest = linear;
    for (uint axis = rank; axis > 0u; axis--) {
        uint extent = uint(meta[ext_at + axis - 1u]);
        uint next = rest / extent;
        at += int(rest - next * extent) * int(meta[str_at + axis - 1u]);
        rest = next;
    }
    return at;
}

/* The fold, and the only kernel that matters for throughput.
 *
 * `width` threads share one segment and a group of 256 therefore covers
 * 256/width of them, so the same code is a group per segment when the reduced
 * axis is long and a lane per segment when it is three elements wide. Both
 * shapes are needed: the first keeps a short output axis from starving the
 * grid, the second keeps a long one from wasting 98% of every group.
 *
 * `flags` bit 0 says the kept axes are a dense run and bit 1 says the reduced
 * ones are, which replaces a walk with a multiply. Bit 2 additionally says the
 * reduced run is unit-strided, a multiple of four long, and sixteen-byte
 * aligned for every segment, which is the float4 path.
 *
 * `chunks` splits one segment across several groups; the caller runs this a
 * second time over the partials to join them. */
kernel void nr_fold(device float* dst [[buffer(0)]],
                    device const float* src [[buffer(1)]],
                    device const float* center [[buffer(2)]],
                    device const float* meta [[buffer(3)]],
                    constant uint& segments [[buffer(4)]],
                    constant uint& n [[buffer(5)]],
                    constant uint& op [[buffer(6)]],
                    constant uint& skip_nan [[buffer(7)]],
                    constant uint& flags [[buffer(8)]],
                    constant uint& base [[buffer(9)]],
                    constant uint& per [[buffer(10)]],
                    constant uint& chunks [[buffer(11)]],
                    constant uint& width [[buffer(12)]],
                    uint gid [[threadgroup_position_in_grid]],
                    uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[ND_GROUP];
    uint block = gid / chunks;
    uint chunk = gid - block * chunks;
    uint sub = lid / width;
    uint slot = lid - sub * width;
    uint seg = block * (ND_GROUP / width) + sub;

    float ident = nr_identity(op, skip_nan);
    float mine = ident;
    /* No early return: every thread has to reach the barriers below, so a
     * segment past the end contributes the identity instead of leaving. */
    if (seg < segments) {
        uint outer_rank = uint(meta[0]);
        uint inner_rank = uint(meta[1]);
        int off = int(base);
        if ((flags & 1u) != 0u) off += int(meta[2]) * int(seg);
        else off += nr_walk(meta, 4u, 4u + outer_rank, outer_rank, seg);
        float c = (op == NR_CENTERED) ? center[seg] : 0.0f;
        uint start = chunk * per;
        uint stop = min(n, start + per);
        if ((flags & 4u) != 0u) {
            device const float4* row = (device const float4*)(src + off);
            float4 acc = float4(ident);
            uint last = stop >> 2u;
            for (uint j = (start >> 2u) + slot; j < last; j += width)
                acc = nr_combine4(acc, nr_ingest4(row[j], c, op, skip_nan), op, skip_nan);
            mine = nr_combine(
                nr_combine(acc.x, acc.y, op, skip_nan),
                nr_combine(acc.z, acc.w, op, skip_nan),
                op,
                skip_nan
            );
        } else if ((flags & 2u) != 0u) {
            int step = int(meta[3]);
            for (uint k = start + slot; k < stop; k += width)
                mine = nr_combine(
                    mine, nr_ingest(src[off + int(k) * step], c, op, skip_nan), op, skip_nan);
        } else {
            uint ext = 4u + 2u * uint(meta[0]);
            for (uint k = start + slot; k < stop; k += width) {
                int at = off + nr_walk(meta, ext, ext + inner_rank, inner_rank, k);
                mine = nr_combine(mine, nr_ingest(src[at], c, op, skip_nan), op, skip_nan);
            }
        }
    }

    scratch[lid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half = width >> 1u; half > 0u; half >>= 1u) {
        if (slot < half) scratch[lid] = nr_combine(scratch[lid], scratch[lid + half], op, skip_nan);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (slot == 0u && seg < segments) dst[seg * chunks + chunk] = scratch[lid];
}

/* argmin and argmax, in the same width/chunk shape as the fold.
 *
 * No float4 path: four lanes need four candidate indices and the compare
 * ladder that picks between them costs more than the wider load saves.
 *
 * Three things are reduced at once. The best key -- the value, negated for
 * argmin so one comparison serves both -- the lowest index carrying it, and
 * the lowest index of any NaN. The last is separate because numpy's argmax
 * answers with the first NaN when one is present, and no float key sorts
 * above every other one including +inf. `skip_nan` switches that off, which
 * is all `nanargmax` is. */
kernel void nr_arg(device float* index [[buffer(0)]],
                   device float* value [[buffer(1)]],
                   device const float* src [[buffer(2)]],
                   device const float* meta [[buffer(3)]],
                   constant uint& segments [[buffer(4)]],
                   constant uint& n [[buffer(5)]],
                   constant uint& smallest [[buffer(6)]],
                   constant uint& skip_nan [[buffer(7)]],
                   constant uint& flags [[buffer(8)]],
                   constant uint& base [[buffer(9)]],
                   constant uint& per [[buffer(10)]],
                   constant uint& chunks [[buffer(11)]],
                   constant uint& width [[buffer(12)]],
                   uint gid [[threadgroup_position_in_grid]],
                   uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float keys[ND_GROUP];
    threadgroup uint spots[ND_GROUP];
    threadgroup uint nans[ND_GROUP];
    uint block = gid / chunks;
    uint chunk = gid - block * chunks;
    uint sub = lid / width;
    uint slot = lid - sub * width;
    uint seg = block * (ND_GROUP / width) + sub;

    float peak = -nr_inf();
    uint best = 0xffffffffu;
    uint nan_at = 0xffffffffu;
    if (seg < segments) {
        uint outer_rank = uint(meta[0]);
        uint inner_rank = uint(meta[1]);
        int off = int(base);
        if ((flags & 1u) != 0u) off += int(meta[2]) * int(seg);
        else off += nr_walk(meta, 4u, 4u + outer_rank, outer_rank, seg);
        int step = int(meta[3]);
        uint ext = 4u + 2u * outer_rank;
        uint start = chunk * per;
        uint stop = min(n, start + per);
        for (uint k = start + slot; k < stop; k += width) {
            int at = (flags & 2u) != 0u
                ? off + int(k) * step
                : off + nr_walk(meta, ext, ext + inner_rank, inner_rank, k);
            float v = src[at];
            if (v != v) {
                if (skip_nan == 0u && k < nan_at) nan_at = k;
                continue;
            }
            float key = smallest != 0u ? -v : v;
            /* Strictly greater, so a tie leaves `best` at the earlier index. */
            if (key > peak) { peak = key; best = k; }
        }
    }

    keys[lid] = peak;
    spots[lid] = best;
    nans[lid] = nan_at;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half = width >> 1u; half > 0u; half >>= 1u) {
        if (slot < half) {
            uint other = lid + half;
            bool take = keys[other] > keys[lid]
                || (keys[other] == keys[lid] && spots[other] < spots[lid]);
            if (take) { keys[lid] = keys[other]; spots[lid] = spots[other]; }
            nans[lid] = min(nans[lid], nans[other]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (slot != 0u || seg >= segments) return;
    uint at = seg * chunks + chunk;
    if (nans[lid] != 0xffffffffu) {
        index[at] = float(nans[lid]);
        value[at] = nr_nan();
    } else if (spots[lid] == 0xffffffffu) {
        /* Nothing to choose from: an empty chunk, or a slice the nan flag
         * emptied. -1 and not a huge sentinel, because 0xffffffff is not a
         * float32 and would come back one larger than it went in. */
        index[at] = -1.0f;
        value[at] = -nr_inf();
    } else {
        index[at] = float(spots[lid]);
        value[at] = keys[lid];
    }
}

/* Join the per-chunk winners. One thread a segment and a serial walk of the
 * chunks: there are at most 1024 of them and the pass reads 1024 floats, so
 * anything cleverer costs more to launch than to run. */
kernel void nr_arg_finish(device float* dst [[buffer(0)]],
                          device const float* index [[buffer(1)]],
                          device const float* value [[buffer(2)]],
                          constant uint& segments [[buffer(3)]],
                          constant uint& chunks [[buffer(4)]],
                          uint seg [[thread_position_in_grid]]) {
    if (seg >= segments) return;
    float peak = 0.0f;
    int best = -1;
    int first_nan = -1;
    bool seen = false;
    for (uint c = 0u; c < chunks; c++) {
        uint at = seg * chunks + c;
        float v = value[at];
        int i = int(index[at]);
        if (v != v) {
            if (first_nan < 0 || i < first_nan) first_nan = i;
            continue;
        }
        if (i < 0) continue;
        if (!seen || v > peak || (v == peak && i < best)) { peak = v; best = i; seen = true; }
    }
    dst[seg] = first_nan >= 0 ? float(first_nan) : float(best);
}

/* sqrt in place, which is all that separates a standard deviation from a
 * variance. Its own kernel rather than a `ufunc` import: `reduce` and `ufunc`
 * are the same layer and the package forbids a sideways dependency. */
kernel void nr_root(device float* values [[buffer(0)]],
                    constant uint& count [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    values[id] = sqrt(values[id]);
}

/* The elementwise half of `allclose` and `array_equal`, through the shared
 * layout buffer so both sides broadcast. `exact` skips the tolerance, and an
 * infinity is only ever close to the same infinity -- `|inf - inf|` is NaN and
 * would otherwise report every pair of them as far apart. */
kernel void nr_close(device float* dst [[buffer(0)]],
                     device const float* a [[buffer(1)]],
                     device const float* b [[buffer(2)]],
                     device const float* meta [[buffer(3)]],
                     constant uint& count [[buffer(4)]],
                     constant uint& exact [[buffer(5)]],
                     constant uint& rtol_bits [[buffer(6)]],
                     constant uint& atol_bits [[buffer(7)]],
                     constant uint& equal_nan [[buffer(8)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float x = a[nd_offset(meta, 1u, id)];
    float y = b[nd_offset(meta, 2u, id)];
    bool ok;
    if (x != x || y != y) ok = equal_nan != 0u && x != x && y != y;
    else if (exact != 0u || isinf(x) || isinf(y)) ok = x == y;
    else ok = fabs(x - y) <= as_type<float>(atol_bits) + as_type<float>(rtol_bits) * fabs(y);
    dst[nd_offset(meta, 0u, id)] = ok ? 1.0f : 0.0f;
}

/* Pass one of the scan: every chunk scanned as if it started from the
 * identity, plus the chunk's total for pass two.
 *
 * Each thread owns a contiguous run so the serial half of the scan stays in
 * registers, then a Hillis-Steele over the 256 thread totals turns those into
 * bases. Hillis-Steele and not `simd_prefix_exclusive_sum`, which exists only
 * for sum and product: cumulative max and min need the same ladder. */
kernel void nr_scan_chunk(device float* dst [[buffer(0)]],
                          device const float* src [[buffer(1)]],
                          device const float* meta [[buffer(2)]],
                          device float* partials [[buffer(3)]],
                          constant uint& segments [[buffer(4)]],
                          constant uint& n [[buffer(5)]],
                          constant uint& op [[buffer(6)]],
                          constant uint& per [[buffer(7)]],
                          constant uint& chunks [[buffer(8)]],
                          constant uint& src_base [[buffer(9)]],
                          uint gid [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[ND_GROUP];
    uint seg = gid / chunks;
    uint chunk = gid - seg * chunks;
    /* Every thread of the group shares `seg`, so leaving together is safe. */
    if (seg >= segments) return;

    uint rank = uint(meta[0]);
    int sstep = int(meta[1]);
    int dstep = int(meta[2]);
    int soff = int(src_base) + nr_walk(meta, 3u, 3u + rank, rank, seg);
    int doff = nr_walk(meta, 3u, 3u + 2u * rank, rank, seg);

    uint start = chunk * per;
    uint stop = min(n, start + per);
    uint each = (per + ND_GROUP - 1u) / ND_GROUP;
    uint from = min(stop, start + lid * each);
    uint to = min(stop, from + each);

    float ident = nr_identity(op, 0u);
    float mine = ident;
    for (uint k = from; k < to; k++) mine = nr_combine(mine, src[soff + int(k) * sstep], op, 0u);

    scratch[lid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint reach = 1u; reach < ND_GROUP; reach <<= 1u) {
        float other = lid >= reach ? scratch[lid - reach] : ident;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scratch[lid] = nr_combine(scratch[lid], other, op, 0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float running = lid == 0u ? ident : scratch[lid - 1u];
    if (lid == 0u && chunks > 1u) partials[seg * chunks + chunk] = scratch[ND_GROUP - 1u];

    for (uint k = from; k < to; k++) {
        running = nr_combine(running, src[soff + int(k) * sstep], op, 0u);
        dst[doff + int(k) * dstep] = running;
    }
}

/* Pass two: one group a segment turns that segment's chunk totals into the
 * base each chunk starts from. Serial in the chunk count, which is capped at
 * 1024 for exactly this reason. */
kernel void nr_scan_bases(device float* partials [[buffer(0)]],
                          constant uint& chunks [[buffer(1)]],
                          constant uint& op [[buffer(2)]],
                          uint seg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[ND_GROUP];
    uint each = (chunks + ND_GROUP - 1u) / ND_GROUP;
    uint from = min(chunks, lid * each);
    uint to = min(chunks, from + each);
    float ident = nr_identity(op, 0u);
    float mine = ident;
    for (uint k = from; k < to; k++) mine = nr_combine(mine, partials[seg * chunks + k], op, 0u);

    scratch[lid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint reach = 1u; reach < ND_GROUP; reach <<= 1u) {
        float other = lid >= reach ? scratch[lid - reach] : ident;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scratch[lid] = nr_combine(scratch[lid], other, op, 0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float running = lid == 0u ? ident : scratch[lid - 1u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = from; k < to; k++) {
        float held = partials[seg * chunks + k];
        partials[seg * chunks + k] = running;
        running = nr_combine(running, held, op, 0u);
    }
}

/* Pass three: fold each chunk's base back into the values pass one wrote.
 * A thread an element, because there is one of those per output and the
 * output is the whole array. */
kernel void nr_scan_apply(device float* dst [[buffer(0)]],
                          device const float* meta [[buffer(1)]],
                          device const float* partials [[buffer(2)]],
                          constant uint& segments [[buffer(3)]],
                          constant uint& n [[buffer(4)]],
                          constant uint& op [[buffer(5)]],
                          constant uint& per [[buffer(6)]],
                          constant uint& chunks [[buffer(7)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= segments * n) return;
    uint seg = id / n;
    uint k = id - seg * n;
    uint chunk = k / per;
    if (chunk == 0u) return;
    uint rank = uint(meta[0]);
    int dstep = int(meta[2]);
    int doff = nr_walk(meta, 3u, 3u + 2u * rank, rank, seg);
    int at = doff + int(k) * dstep;
    dst[at] = nr_combine(partials[seg * chunks + chunk], dst[at], op, 0u);
}
"""

#: Groups a chunked fold or scan aims for. 1024 fills every Apple GPU this
#: runs on, and it is also the most partials the single finish dispatch wants
#: to walk — the two bounds want the same number, so there is one constant.
const _TARGET_GROUPS = 1024

#: Elements below which splitting a segment across groups costs more in
#: launches and partials than it wins in parallelism.
const _CHUNK_FLOOR = 2048

fn _larger(a: int, b: int) -> int { return if a > b { a } else { b } }

fn _smaller(a: int, b: int) -> int { return if a < b { a } else { b } }

fn _magnitude(value: int) -> int { return if value < 0 { -value } else { value } }

#: The axes a call names, normalised: `null` is every axis, an int is one, and
#: a list or tuple is several. numpy's `axis=()` — reduce nothing — arrives
#: here as an empty list and folds each element with itself, which is a copy.
fn _axes_of(axis: any, ndim: int) -> list[int] {
    if axis is null { return [at for at in 0..ndim] }
    if axis is int { return [normalize_axis(axis, ndim)] }
    if axis is list { return normalize_axes([at for at in axis], ndim) }
    if axis is tuple { return normalize_axes([at for at in axis], ndim) }
    throw TypeError(f"an axis is an int, a list of ints or null, not {type_of(axis)}")
}

#: The shape a reduction over `axes` leaves behind.
fn _result_shape(shape: list[int], axes: list[int], keepdims: bool) -> list[int] {
    var reduced = [false for _axis in shape]
    for at in axes { reduced[at] = true }
    var out: list[int] = []
    for axis in 0..shape.len() {
        if not reduced[axis] {
            out.push(shape[axis])
        } elif keepdims {
            out.push(1)
        }
    }
    return out
}

#: Elements one output of a reduction over `axes` folds.
fn _reduced_count(shape: list[int], axes: list[int]) -> int {
    var total = 1
    for at in axes { total *= shape[at] }
    return total
}

#: The dtype a total or a product accumulates in.
#:
#: numpy widens an integer input to the platform integer and leaves a float
#: alone, and since `dtype` names the accumulator it also names the result.
fn _sum_dtype(source: int, dtype: int) -> int {
    if dtype >= 0 { return dtype }
    if dtype_is_integer(source) { return INT64 }
    return source
}

#: The dtype an average accumulates in: never an integer one, because the
#: answer is a ratio.
fn _float_dtype(source: int, dtype: int) -> int {
    if dtype >= 0 { return dtype }
    if dtype_is_integer(source) { return FLOAT32 }
    return source
}

#: `a` read through the accumulator dtype the caller named.
#:
#: Only when that rounds — accumulating float data in an integer dtype is the
#: one case where numpy truncates on the way in. Widening a `BOOL` to an
#: `INT64` changes no bits and is not worth a copy.
fn _as_accumulator(a: NDArray, dtype: int) -> NDArray {
    if dtype < 0 or dtype_is_integer(a.dtype) or not dtype_is_integer(dtype) { return a }
    return a.astype(dtype)
}

#: Reduced axes in falling stride order, so the one that moves fastest ends up
#: innermost and neighbouring threads read neighbouring slots.
#:
#: An insertion sort: there are as many axes as the array has dimensions.
fn _by_falling_stride(axes: list[int], strides: list[int]) -> list[int] {
    var out = [at for at in axes]
    for index in 1..out.len() {
        let axis = out[index]
        var at = index
        while at > 0 and _magnitude(strides[out[at - 1]]) < _magnitude(strides[axis]) {
            out[at] = out[at - 1]
            at -= 1
        }
        out[at] = axis
    }
    return out
}

#: How a fold is spread over the grid: the two extents, the access shortcuts
#: the layout allows, and the metadata block the kernel walks.
class _Plan {
    #: Outputs, one per element of the kept axes.
    pub let segments: int
    #: Elements each output folds.
    pub let n: int
    #: Bit 0 the kept axes are a dense run, bit 1 the reduced axes are, bit 2
    #: the reduced run also takes float4 loads.
    pub let flags: int
    #: Threads one segment gets. A power of two dividing `GROUP`.
    pub let width: int
    #: Elements one chunk of a segment covers.
    pub let per: int
    #: Chunks a segment is split into. More than one asks for a finish pass.
    pub let chunks: int
    #: Extents and strides of both halves, as `nr_walk` reads them.
    pub let meta: Buffer

    pub fn init(
        self,
        segments: int,
        n: int,
        flags: int,
        width: int,
        per: int,
        chunks: int,
        meta: Buffer
    ) {
        self.segments = segments
        self.n = n
        self.flags = flags
        self.width = width
        self.per = per
        self.chunks = chunks
        self.meta = meta
    }

    #: Threadgroups the first dispatch runs.
    pub fn groups(self) -> int {
        let each = GROUP // self.width
        return (self.segments + each - 1) // each * self.chunks
    }
}

#: Chunks worth splitting a segment into, given how many segments there
#: already are. Capped so `segments * chunks` never passes `_TARGET_GROUPS`
#: and the finish pass therefore stays one dispatch over a small buffer.
fn _chunks_for(segments: int, n: int) -> int {
    if segments >= _TARGET_GROUPS { return 1 }
    let by_grid = _TARGET_GROUPS // _larger(1, segments)
    let by_work = n // _CHUNK_FLOOR
    return _larger(1, _smaller(by_grid, by_work))
}

#: Split `a`'s axes into the kept half and the reduced half and work out how
#: the grid should cover them.
#:
#: `sort_inner` is false for the arg-folds, whose answer is an index into the
#: reduced block: that index has to decompose in the caller's axis order, so
#: they give up the coalescing the stride sort would buy.
fn _plan(a: NDArray, axes: list[int], allow_quad: bool, sort_inner: bool) -> _Plan {
    let rank = a.ndim()
    var reduced = [false for _axis in 0..rank]
    for at in axes { reduced[at] = true }

    var outer_shape: list[int] = []
    var outer_strides: list[int] = []
    for axis in 0..rank {
        if reduced[axis] { continue }
        outer_shape.push(a.shape[axis])
        outer_strides.push(a.strides[axis])
    }
    let order = if sort_inner { _by_falling_stride(axes, a.strides) } else { axes }
    var inner_shape: list[int] = []
    var inner_strides: list[int] = []
    for axis in order {
        inner_shape.push(a.shape[axis])
        inner_strides.push(a.strides[axis])
    }

    var flags = 0
    var outer_step = 0
    var inner_step = 0
    if outer_shape.len() == 0 {
        flags = flags | 1
    } else {
        outer_step = outer_strides[outer_strides.len() - 1]
        if is_contiguous(outer_shape, outer_strides, outer_step) {
            flags = flags | 1
        } else {
            outer_step = 0
        }
    }
    if inner_shape.len() == 0 {
        flags = flags | 2
    } else {
        inner_step = inner_strides[inner_strides.len() - 1]
        if is_contiguous(inner_shape, inner_strides, inner_step) {
            flags = flags | 2
        } else {
            inner_step = 0
        }
    }

    let segments = element_count(outer_shape)
    let n = element_count(inner_shape)
    #: A float4 load needs the run to be unit-strided and every segment's first
    #: slot to be sixteen-byte aligned, which is a property of the offset and
    #: of the kept strides, both known here.
    var quad = allow_quad and flags & 2 != 0 and inner_step == 1 and n % 4 == 0
    if quad {
        var aligned = a.offset % 4 == 0
        for stride in outer_strides { if stride % 4 != 0 { aligned = false } }
        quad = aligned
    }
    if quad { flags = flags | 4 }

    let units = if quad { n // 4 } else { n }
    var width = 1
    while width < units and width < GROUP { width *= 2 }
    var chunks = _chunks_for(segments, n)
    var per = n
    if chunks > 1 {
        width = GROUP
        per = (n + chunks - 1) // chunks
        #: A chunk boundary that is not a multiple of four would unalign every
        #: float4 load after the first chunk.
        if quad { per = (per + 3) // 4 * 4 }
        chunks = (n + per - 1) // per
    }

    var values = [
        float(outer_shape.len()),
        float(inner_shape.len()),
        float(outer_step),
        float(inner_step),
    ]
    for extent in outer_shape { values.push(float(extent)) }
    for stride in outer_strides { values.push(float(stride)) }
    for extent in inner_shape { values.push(float(extent)) }
    for stride in inner_strides { values.push(float(stride)) }
    let meta = Buffer(values.len())
    meta.upload(values)
    return _Plan(segments, n, flags, width, per, chunks, meta)
}

#: The four-float metadata a fold over an already-dense `[segments, n]` block
#: reads: no walk, both steps known.
fn _flat_meta(outer_step: int, inner_step: int) -> Buffer {
    let meta = Buffer(4)
    meta.upload([0.0, 0.0, float(outer_step), float(inner_step)])
    return meta
}

#: The op that joins partials, which is not always the op that made them: a
#: count and a squared deviation are both totalled, whatever produced them.
fn _finish_op(op: int) -> int {
    if op == OP_NONZERO or op == OP_FINITE or op == OP_CENTERED { return OP_SUM }
    return op
}

#: What a fold over nothing answers, so an empty axis needs no dispatch.
fn _identity_of(op: int, skip_nan: bool) -> float {
    if op == OP_PROD or op == OP_ALL { return 1.0 }
    if op == OP_MIN or op == OP_MAX {
        if skip_nan { return 0.0 / 0.0 }
        let huge = 1.0e308 * 10.0
        return if op == OP_MIN { huge } else { -huge }
    }
    return 0.0
}

#: Run `op` over `a`'s `axes` into `dst`, which holds one float per segment.
fn _fold_into(
    dst: Buffer,
    a: NDArray,
    axes: list[int],
    op: int,
    skip_nan: int,
    center: Buffer?
) -> void {
    let plan = _plan(a, axes, true, true)
    defer { plan.meta.free() }
    #: An unused centre still has to be bound, and the source is the one
    #: read-only allocation already resident for this dispatch.
    let gate = center ?? a.storage
    let fold = kernel_of(SOURCE, "nr_fold")
    if plan.chunks == 1 {
        fold.dispatch_async(
            [dst, a.storage, gate, plan.meta],
            plan.groups() * GROUP,
            [plan.segments, plan.n, op, skip_nan, plan.flags, a.offset, plan.per, 1, plan.width],
            GROUP
        )
        return
    }
    let partials = Buffer(plan.segments * plan.chunks)
    let flat = _flat_meta(plan.chunks, 1)
    defer { partials.free() }
    defer { flat.free() }
    fold.dispatch_async(
        [partials, a.storage, gate, plan.meta],
        plan.groups() * GROUP,
        [
            plan.segments,
            plan.n,
            op,
            skip_nan,
            plan.flags,
            a.offset,
            plan.per,
            plan.chunks,
            plan.width,
        ],
        GROUP
    )
    fold.dispatch_async(
        [dst, partials, partials, flat],
        plan.segments * GROUP,
        [plan.segments, plan.chunks, _finish_op(op), skip_nan, 3, 0, plan.chunks, 1, GROUP],
        GROUP
    )
}

#: The whole of a plain reduction: shape it, hand the empty case its identity,
#: and dispatch.
fn _reduce(
    a: NDArray,
    axis: any,
    keepdims: bool,
    dtype: int,
    result_dtype: int,
    op: int,
    skip_nan: int,
    what: str
) -> NDArray {
    require_real(a, what)
    let axes = _axes_of(axis, a.ndim())
    let out = NDArray(_result_shape(a.shape, axes, keepdims), result_dtype)
    if out.size() == 0 { return out }
    let n = _reduced_count(a.shape, axes)
    if n == 0 {
        if op == OP_MIN or op == OP_MAX {
            throw ValueError(f"{what} of a zero-size axis has no answer")
        }
        fill(out, _identity_of(op, skip_nan != 0))
        return out
    }
    let work = _as_accumulator(a, dtype)
    defer { if not work == a { work.free() } }
    _fold_into(out.storage, work, axes, op, skip_nan, null)
    return out
}

#: Total along `axis`, or over everything when it is null.
#:
#: The accumulator is `dtype` if named, `INT64` for any integer or boolean
#: input, and the input's dtype otherwise — numpy's rule, which is why summing
#: a `BOOL` array counts rather than saturating at one.
pub fn sum(a: NDArray, axis: any = null, keepdims: bool = false, dtype: int = -1) -> NDArray {
    return _reduce(a, axis, keepdims, dtype, _sum_dtype(a.dtype, dtype), OP_SUM, 0, "sum")
}

#: Product along `axis`.
pub fn prod(a: NDArray, axis: any = null, keepdims: bool = false, dtype: int = -1) -> NDArray {
    return _reduce(a, axis, keepdims, dtype, _sum_dtype(a.dtype, dtype), OP_PROD, 0, "prod")
}

#: Smallest value along `axis`. NaN propagates, as numpy's does; `nanmin`
#: is the one that steps over it.
pub fn amin(a: NDArray, axis: any = null, keepdims: bool = false, dtype: int = -1) -> NDArray {
    return _reduce(a, axis, keepdims, dtype, if dtype >= 0 {
        dtype
    } else {
        a.dtype
    }, OP_MIN, 0, "min")
}

#: Largest value along `axis`.
pub fn amax(a: NDArray, axis: any = null, keepdims: bool = false, dtype: int = -1) -> NDArray {
    return _reduce(a, axis, keepdims, dtype, if dtype >= 0 {
        dtype
    } else {
        a.dtype
    }, OP_MAX, 0, "max")
}

#: `amin` under numpy's other name.
pub fn min(a: NDArray, axis: any = null, keepdims: bool = false, dtype: int = -1) -> NDArray {
    return amin(a, axis, keepdims, dtype)
}

#: `amax` under numpy's other name.
pub fn max(a: NDArray, axis: any = null, keepdims: bool = false, dtype: int = -1) -> NDArray {
    return amax(a, axis, keepdims, dtype)
}

#: Whether any element along `axis` is non-zero. NaN counts as true, as it
#: does in numpy.
pub fn any(a: NDArray, axis: any = null, keepdims: bool = false) -> NDArray {
    return _reduce(a, axis, keepdims, -1, BOOL, OP_ANY, 0, "any")
}

#: Whether every element along `axis` is non-zero.
pub fn all(a: NDArray, axis: any = null, keepdims: bool = false) -> NDArray {
    return _reduce(a, axis, keepdims, -1, BOOL, OP_ALL, 0, "all")
}

#: How many elements along `axis` are non-zero.
pub fn count_nonzero(a: NDArray, axis: any = null, keepdims: bool = false) -> NDArray {
    return _reduce(a, axis, keepdims, -1, INT64, OP_NONZERO, 0, "count_nonzero")
}

#: Largest minus smallest along `axis` — numpy's peak to peak.
pub fn ptp(a: NDArray, axis: any = null, keepdims: bool = false) -> NDArray {
    let high = amax(a, axis, keepdims)
    let low = amin(a, axis, keepdims)
    defer { high.free() }
    defer { low.free() }
    return binary_arith(high, low, 1)
}

#: Arithmetic mean along `axis`, always in a float dtype.
pub fn mean(a: NDArray, axis: any = null, keepdims: bool = false, dtype: int = -1) -> NDArray {
    let target = _float_dtype(a.dtype, dtype)
    let total = _reduce(a, axis, keepdims, dtype, target, OP_SUM, 0, "mean")
    defer { total.free() }
    let n = _reduced_count(a.shape, _axes_of(axis, a.ndim()))
    #: An empty mean is 0/0, which numpy also answers with NaN.
    return binary_arith(total, float(n), 3)
}

#: Total along `axis` with the NaNs read as zero. An all-NaN slice therefore
#: totals zero, which is what numpy answers too.
pub fn nansum(a: NDArray, axis: any = null, keepdims: bool = false, dtype: int = -1) -> NDArray {
    return _reduce(a, axis, keepdims, dtype, _sum_dtype(a.dtype, dtype), OP_SUM, 1, "nansum")
}

#: Smallest non-NaN value along `axis`, and NaN where a slice held nothing
#: else. That falls out of the identity rather than costing a counting pass:
#: the fold starts at NaN and the combine prefers a real operand.
pub fn nanmin(a: NDArray, axis: any = null, keepdims: bool = false, dtype: int = -1) -> NDArray {
    return _reduce(a, axis, keepdims, dtype, if dtype >= 0 {
        dtype
    } else {
        a.dtype
    }, OP_MIN, 1, "nanmin")
}

#: Largest non-NaN value along `axis`, NaN for an all-NaN slice.
pub fn nanmax(a: NDArray, axis: any = null, keepdims: bool = false, dtype: int = -1) -> NDArray {
    return _reduce(a, axis, keepdims, dtype, if dtype >= 0 {
        dtype
    } else {
        a.dtype
    }, OP_MAX, 1, "nanmax")
}

#: How many elements along `axis` are not NaN — the divisor the nan-aware
#: averages need, and useful on its own.
fn _nan_count(a: NDArray, axis: any, keepdims: bool) -> NDArray {
    return _reduce(a, axis, keepdims, -1, FLOAT32, OP_FINITE, 0, "nanmean")
}

#: Mean of the non-NaN elements along `axis`. An all-NaN slice divides zero by
#: zero and comes back NaN, as numpy's does.
pub fn nanmean(a: NDArray, axis: any = null, keepdims: bool = false, dtype: int = -1) -> NDArray {
    let total = nansum(a, axis, keepdims, _float_dtype(a.dtype, dtype))
    let tally = _nan_count(a, axis, keepdims)
    defer { total.free() }
    defer { tally.free() }
    return binary_arith(total, tally, 3)
}

#: The shared body of `variance` and `nanvar`.
#:
#: Two passes over the data: the mean, then the total squared deviation from
#: it. The one-pass `E[x^2] - E[x]^2` needs a single read but loses every
#: significant digit once the mean dwarfs the spread, and can answer with a
#: negative variance; the second read is the cheaper mistake.
fn _variance(
    a: NDArray,
    axis: any,
    keepdims: bool,
    dtype: int,
    ddof: int,
    skip_nan: int,
    what: str
) -> NDArray {
    require_real(a, what)
    let axes = _axes_of(axis, a.ndim())
    let target = _float_dtype(a.dtype, dtype)
    #: `keepdims` on the centre so its storage is exactly one float per
    #: segment, in segment order, which is what the kernel indexes it as.
    let centre = if skip_nan != 0 {
        nanmean(a, axis, true, dtype)
    } else {
        mean(a, axis, true, dtype)
    }
    defer { centre.free() }

    let out = NDArray(_result_shape(a.shape, axes, keepdims), target)
    if out.size() == 0 { return out }
    let n = _reduced_count(a.shape, axes)
    if n == 0 {
        fill(out, 0.0 / 0.0)
        return out
    }
    let work = _as_accumulator(a, dtype)
    defer { if not work == a { work.free() } }
    _fold_into(out.storage, work, axes, OP_CENTERED, skip_nan, centre.storage)

    if skip_nan == 0 {
        let divisor = float(n - ddof)
        return binary_arith(out, divisor, 3)
    }
    let tally = _nan_count(a, axis, keepdims)
    defer { tally.free() }
    let divisor = binary_arith(tally, float(ddof), 1)
    defer { divisor.free() }
    return binary_arith(out, divisor, 3)
}

#: Variance along `axis`, dividing by `n - ddof`.
#:
#: Spelled `variance` and not `var`, which numpy uses: `var` is this
#: language's keyword for a mutable binding and cannot name a function.
#: `nanvar` keeps numpy's name because that one parses.
pub fn variance(
    a: NDArray,
    axis: any = null,
    keepdims: bool = false,
    dtype: int = -1,
    ddof: int = 0
) -> NDArray {
    let out = _variance(a, axis, keepdims, dtype, ddof, 0, "var")
    return out
}

#: Variance of the non-NaN elements, dividing by their count less `ddof`.
pub fn nanvar(
    a: NDArray,
    axis: any = null,
    keepdims: bool = false,
    dtype: int = -1,
    ddof: int = 0
) -> NDArray {
    return _variance(a, axis, keepdims, dtype, ddof, 1, "nanvar")
}

fn _root_in_place(a: NDArray) -> NDArray {
    let count = a.size()
    if count == 0 { return a }
    kernel_of(SOURCE, "nr_root").dispatch_async([a.storage], count, [count])
    return a
}

#: Standard deviation along `axis` — the root of `variance`, taken in place so
#: the square roots cost one pass over the outputs and no second allocation.
pub fn std(
    a: NDArray,
    axis: any = null,
    keepdims: bool = false,
    dtype: int = -1,
    ddof: int = 0
) -> NDArray {
    return _root_in_place(variance(a, axis, keepdims, dtype, ddof))
}

#: Standard deviation of the non-NaN elements.
pub fn nanstd(
    a: NDArray,
    axis: any = null,
    keepdims: bool = false,
    dtype: int = -1,
    ddof: int = 0
) -> NDArray {
    return _root_in_place(nanvar(a, axis, keepdims, dtype, ddof))
}

#: Index of the extreme element along `axis`, ties going to the lowest index.
#:
#: With several axes, or with none named, the index is into the C-order
#: flattening of the reduced block — which for a single axis is just the
#: position along it, and for `axis = null` is the flat index numpy returns.
fn _argfold(
    a: NDArray,
    axis: any,
    keepdims: bool,
    smallest: int,
    skip_nan: int,
    what: str
) -> NDArray {
    require_real(a, what)
    let axes = _axes_of(axis, a.ndim())
    let out = NDArray(_result_shape(a.shape, axes, keepdims), INT64)
    if out.size() == 0 { return out }
    let n = _reduced_count(a.shape, axes)
    if n == 0 { throw ValueError(f"{what} of a zero-size axis has no answer") }

    let plan = _plan(a, axes, false, false)
    defer { plan.meta.free() }
    let values = Buffer(plan.segments * plan.chunks)
    defer { values.free() }
    let scalars = [
        plan.segments,
        plan.n,
        smallest,
        skip_nan,
        plan.flags,
        a.offset,
        plan.per,
        plan.chunks,
        plan.width,
    ]
    if plan.chunks == 1 {
        kernel_of(SOURCE, "nr_arg").dispatch_async(
            [out.storage, values, a.storage, plan.meta],
            plan.groups() * GROUP,
            scalars,
            GROUP
        )
        return out
    }
    let spots = Buffer(plan.segments * plan.chunks)
    defer { spots.free() }
    kernel_of(SOURCE, "nr_arg").dispatch_async(
        [spots, values, a.storage, plan.meta],
        plan.groups() * GROUP,
        scalars,
        GROUP
    )
    kernel_of(SOURCE, "nr_arg_finish").dispatch_async(
        [out.storage, spots, values],
        plan.segments,
        [plan.segments, plan.chunks]
    )
    return out
}

#: Index of the smallest element along `axis`. A NaN anywhere in the slice
#: wins, and the answer is the first of them, which is numpy's rule.
pub fn argmin(a: NDArray, axis: any = null, keepdims: bool = false) -> NDArray {
    return _argfold(a, axis, keepdims, 1, 0, "argmin")
}

#: Index of the largest element along `axis`.
pub fn argmax(a: NDArray, axis: any = null, keepdims: bool = false) -> NDArray {
    return _argfold(a, axis, keepdims, 0, 0, "argmax")
}

#: Index of the smallest non-NaN element along `axis`.
#:
#: An all-NaN slice answers -1 rather than raising the way numpy does: knowing
#: the slice was empty means reading the result back, and a synchronise inside
#: a reduction is a price every well-formed call would pay for the one that
#: is not. Callers that owe numpy's `ValueError` test for -1.
pub fn nanargmin(a: NDArray, axis: any = null, keepdims: bool = false) -> NDArray {
    return _argfold(a, axis, keepdims, 1, 1, "nanargmin")
}

#: Index of the largest non-NaN element along `axis`, -1 for an all-NaN slice.
pub fn nanargmax(a: NDArray, axis: any = null, keepdims: bool = false) -> NDArray {
    return _argfold(a, axis, keepdims, 0, 1, "nanargmax")
}

#: Total of a diagonal — `sum` over the axis `diagonal_view` appends.
#:
#: The diagonal is a view whose stride is the sum of the two axes it replaces,
#: so this reads the original allocation and copies nothing.
pub fn trace(
    a: NDArray,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: int = -1
) -> NDArray {
    let line = diagonal_view(a, offset, axis1, axis2)
    return sum(line, -1, false, dtype)
}

#: A cumulative pass along one axis, written through a fresh contiguous copy
#: of the input's shape.
fn _scan(a: NDArray, axis: any, dtype: int, op: int, what: str) -> NDArray {
    require_real(a, what)
    #: `axis = null` scans the flattening, and numpy's answer is 1-d. The
    #: source has to be dense for that view to exist.
    var source = a
    var owned = false
    var at = 0
    if axis is null {
        source = a.contiguous()
        owned = not source == a
        let flat = source.size()
        source = source.view([flat], [source.lanes()], source.offset)
    } else {
        at = normalize_axis(if axis is int { axis } else { _axes_of(axis, a.ndim())[0] }, a.ndim())
    }
    defer { if owned { source.free() } }

    let target = if dtype >= 0 { dtype } else { _sum_dtype(source.dtype, -1) }
    let want = if op == OP_SUM or op == OP_PROD { target } else { source.dtype }
    let out = NDArray(source.shape, want)
    let n = source.shape[at]
    let segments = if n == 0 { 0 } else { source.size() // n }
    if segments == 0 or n == 0 { return out }

    var rank = 0
    var extents: list[float] = []
    var src_strides: list[float] = []
    var dst_strides: list[float] = []
    for axis_at in 0..source.ndim() {
        if axis_at == at { continue }
        rank += 1
        extents.push(float(source.shape[axis_at]))
        src_strides.push(float(source.strides[axis_at]))
        dst_strides.push(float(out.strides[axis_at]))
    }
    var values = [float(rank), float(source.strides[at]), float(out.strides[at])]
    for value in extents { values.push(value) }
    for value in src_strides { values.push(value) }
    for value in dst_strides { values.push(value) }
    let meta = Buffer(values.len())
    defer { meta.free() }
    meta.upload(values)

    var chunks = _chunks_for(segments, n)
    var per = n
    if chunks > 1 {
        per = (n + chunks - 1) // chunks
        chunks = (n + per - 1) // per
    }
    let partials = Buffer(segments * chunks)
    defer { partials.free() }
    kernel_of(SOURCE, "nr_scan_chunk").dispatch_async(
        [out.storage, source.storage, meta, partials],
        segments * chunks * GROUP,
        [segments, n, op, per, chunks, source.offset],
        GROUP
    )
    if chunks == 1 { return out }
    kernel_of(SOURCE, "nr_scan_bases").dispatch_async(
        [partials],
        segments * GROUP,
        [chunks, op],
        GROUP
    )
    kernel_of(SOURCE, "nr_scan_apply").dispatch_async(
        [out.storage, meta, partials],
        segments * n,
        [segments, n, op, per, chunks]
    )
    return out
}

#: Running total along `axis`, or along the flattening when it is null.
#:
#: Three dispatches and not a host loop: each group scans its own chunk, one
#: group a segment scans the chunk totals, and a third pass folds those bases
#: back in — the shape `array.jai`'s compaction scan uses, with the combine
#: left open so `cumprod`, `cummax` and `cummin` are the same code.
pub fn cumsum(a: NDArray, axis: any = null, dtype: int = -1) -> NDArray {
    return _scan(a, axis, dtype, OP_SUM, "cumsum")
}

#: Running product along `axis`.
pub fn cumprod(a: NDArray, axis: any = null, dtype: int = -1) -> NDArray {
    return _scan(a, axis, dtype, OP_PROD, "cumprod")
}

#: Running maximum along `axis`. numpy spells this `maximum.accumulate`.
pub fn cummax(a: NDArray, axis: any = null) -> NDArray {
    return _scan(a, axis, -1, OP_MAX, "cummax")
}

#: Running minimum along `axis`.
pub fn cummin(a: NDArray, axis: any = null) -> NDArray {
    return _scan(a, axis, -1, OP_MIN, "cummin")
}

#: The elementwise agreement `allclose` and `array_equal` reduce.
fn _agreement(
    a: NDArray,
    b: NDArray,
    exact: int,
    rtol: float,
    atol: float,
    equal_nan: bool
) -> NDArray {
    require_real(a, "allclose")
    require_real(b, "allclose")
    let out = NDArray(_broadcast_shape(a, b), BOOL)
    let count = out.size()
    if count == 0 { return out }
    let left = a.broadcast_to(out.shape)
    let right = b.broadcast_to(out.shape)
    let meta = layout_buffer([out, left, right])
    defer { meta.free() }
    kernel_of(SOURCE, "nr_close").dispatch_async([out.storage, left.storage, right.storage, meta], count, [count, exact, f32_bits(rtol), f32_bits(atol), if equal_nan {
        1
    } else {
        0
    }])
    return out
}

fn _broadcast_shape(a: NDArray, b: NDArray) -> list[int] {
    let rank = _larger(a.ndim(), b.ndim())
    var out = [0 for _axis in 0..rank]
    for axis in 0..rank {
        let left = if axis < rank - a.ndim() { 1 } else { a.shape[axis - (rank - a.ndim())] }
        let right = if axis < rank - b.ndim() { 1 } else { b.shape[axis - (rank - b.ndim())] }
        if left != right and left != 1 and right != 1 {
            throw ValueError(f"shapes {a.shape} and {b.shape} do not broadcast together")
        }
        out[axis] = _larger(left, right)
    }
    return out
}

#: Whether every element folds to true. The one synchronise these predicates
#: owe: a `bool` is a host value and there is no way to have one without
#: reading the device.
fn _every(mask: NDArray) -> bool {
    let folded = all(mask)
    defer { folded.free() }
    return folded.item() != 0.0
}

#: Whether `a` and `b` agree everywhere to within `atol + rtol * |b|`.
#:
#: Asymmetric in `a` and `b`, as numpy's is. Infinities have to match sign for
#: sign — the tolerance test on two of them is `|inf - inf|`, which is NaN and
#: would call every pair of infinities far apart.
pub fn allclose(
    a: NDArray,
    b: NDArray,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    equal_nan: bool = false
) -> bool {
    let mask = _agreement(a, b, 0, rtol, atol, equal_nan)
    defer { mask.free() }
    return _every(mask)
}

#: Whether `a` and `b` have the same shape and the same values.
#:
#: A shape mismatch is false rather than an error, which is what makes this
#: usable as a guard. `equal_nan` reads NaN as equal to NaN, which no
#: comparison does on its own.
pub fn array_equal(a: NDArray, b: NDArray, equal_nan: bool = false) -> bool {
    if a.shape != b.shape { return false }
    if a.size() == 0 { return true }
    let mask = _agreement(a, b, 1, 0.0, 0.0, equal_nan)
    defer { mask.free() }
    return _every(mask)
}
