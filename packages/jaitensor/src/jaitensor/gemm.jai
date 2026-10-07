#: Matrix multiplication: one general simdgroup kernel family, measured routing.
#:
#: A single templated Metal kernel covers every shape, both transposes, a fused
#: bias/activation epilogue and split-K. It is instantiated once per tile
#: configuration; `plan_for` decides which instantiation (or MPSGraph) runs a
#: given shape by timing the candidates on the device the program is actually
#: using, and caches the answer on disk.
#:
#: A shape runs the static plan until it has earned a tuning pass: plans are
#: kept per BUCKET of shapes (rows and inner dimension rounded up to a quarter
#: octave, columns exact), and a bucket is tuned only once one of its exact
#: shapes has been asked for `TUNE_MIN_REQUESTS` times and the work it did is
#: estimated to have cost `TUNE_PAYBACK_SECONDS`. `JAITENSOR_AUTOTUNE=0` keeps the static
#: heuristic instead, `JAITENSOR_AUTOTUNE_VERBOSE=1` prints every candidate it
#: timed, `JAITENSOR_GEMM_CACHE` overrides the cache path, and
#: `JAITENSOR_GEMM_PLAN` forces one plan for every shape (`name` or
#: `name:splits`; anything else is a `ValueError`).
#:
#: Switches, each read once and on by default, so every one of them is an A/B
#: in one binary:
#: - `JAITENSOR_GEMM_LAZY_TUNE=0` keys plans on the exact shape and tunes on
#:   the first call, as before buckets existed.
#: - `JAITENSOR_GEMM_PAD=0` keeps MPSGraph off any shape whose stored rows are
#:   not a multiple of four floats, instead of zero-padding a copy.
#: - `JAITENSOR_GEMM_STATIC_DIRECT=0` keeps the old static plan, which only
#:   ever chose a staged kernel.
#: - `JAITENSOR_GEMM_DEEP_SPLITS=0` stops splitting K past sixteen.
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from std.io import Path
from std.os import env, env_or, home_dir
from std.time import Instant

let _LAZY_TUNE = env_or("JAITENSOR_GEMM_LAZY_TUNE", "1") != "0"
let _PAD = env_or("JAITENSOR_GEMM_PAD", "1") != "0"
let _STATIC_DIRECT = env_or("JAITENSOR_GEMM_STATIC_DIRECT", "1") != "0"
let _DEEP_SPLITS = env_or("JAITENSOR_GEMM_DEEP_SPLITS", "1") != "0"

let TEMPLATE = """
kernel void NAME(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* output [[buffer(2)]],
                 device const float* bias [[buffer(3)]],
                 constant uint& rows [[buffer(4)]],
                 constant uint& inner [[buffer(5)]],
                 constant uint& columns [[buffer(6)]],
                 constant uint& trans_a [[buffer(7)]],
                 constant uint& trans_b [[buffer(8)]],
                 constant uint& epilogue [[buffer(9)]],
                 constant uint& splits [[buffer(10)]],
                 constant uint& stride_a [[buffer(11)]],
                 constant uint& stride_b [[buffer(12)]],
                 constant uint& stride_out [[buffer(13)]],
                 uint lid [[thread_index_in_threadgroup]],
                 uint wid [[threadgroup_position_in_grid]],
                 uint sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = _BM_;
    constexpr uint BN = _BN_;
    constexpr uint BK = _BK_;
    constexpr uint WM = _WM_;
    constexpr uint WN = _WN_;
    constexpr uint THREADS = WM * WN * 32u;
    constexpr uint TM = BM / WM / 8u;
    constexpr uint TN = BN / WN / 8u;
    constexpr uint ALOAD = BM * BK / THREADS;
    constexpr uint BLOAD = BK * BN / THREADS;
    constexpr uint SGW = BN / WN;
    constexpr uint STAGE = (BM * BK + BK * BN) * sizeof(_ET_);
    constexpr uint EPILOGUE_BYTES = WM * WN * 8u * SGW * 4u;
    constexpr uint POOL = ((STAGE > EPILOGUE_BYTES ? STAGE : EPILOGUE_BYTES) + 3u) / 4u;

    threadgroup float pool[POOL];
    threadgroup _ET_* As = (threadgroup _ET_*)pool;
    threadgroup _ET_* Bs = As + BM * BK;

    const uint groups_x = (columns + BN - 1u) / BN;
    const uint groups_y = (rows + BM - 1u) / BM;
    const uint tiles = groups_x * groups_y;
    const uint per_item = tiles * splits;
    const uint item = wid / per_item;
    const uint local = wid - item * per_item;
    const uint slice = local / tiles;
    const uint tile = local - slice * tiles;
    a += item * stride_a;
    b += item * stride_b;
    output += item * stride_out;
    const uint tile_row = (tile / groups_x) * BM;
    const uint tile_col = (tile % groups_x) * BN;
    const uint sg_m = sgitg / WN;
    const uint sg_n = sgitg % WN;

    const uint chunk = ((inner + splits - 1u) / splits + BK - 1u) / BK * BK;
    const uint k_begin = slice * chunk;
    const uint k_stop = min(inner, k_begin + chunk);

    simdgroup_matrix<float, 8, 8> acc[TM][TN];
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    float areg[ALOAD];
    float breg[BLOAD];

    if (k_begin < k_stop) {
        for (uint t = 0; t < ALOAD; ++t) {
            uint idx = lid + t * THREADS;
            uint m = tile_row + idx / BK;
            uint kk = k_begin + idx % BK;
            float v = 0.0f;
            if (m < rows && kk < k_stop)
                v = trans_a ? a[kk * rows + m] : a[m * inner + kk];
            areg[t] = v;
        }
        for (uint t = 0; t < BLOAD; ++t) {
            uint idx = lid + t * THREADS;
            uint kk = k_begin + idx / BN;
            uint n = tile_col + idx % BN;
            float v = 0.0f;
            if (n < columns && kk < k_stop)
                v = trans_b ? b[n * inner + kk] : b[kk * columns + n];
            breg[t] = v;
        }

        for (uint k0 = k_begin; k0 < k_stop; k0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = 0; t < ALOAD; ++t) As[lid + t * THREADS] = (_ET_)areg[t];
            for (uint t = 0; t < BLOAD; ++t) Bs[lid + t * THREADS] = (_ET_)breg[t];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint next = k0 + BK;
            if (next < k_stop) {
                for (uint t = 0; t < ALOAD; ++t) {
                    uint idx = lid + t * THREADS;
                    uint m = tile_row + idx / BK;
                    uint kk = next + idx % BK;
                    float v = 0.0f;
                    if (m < rows && kk < k_stop)
                        v = trans_a ? a[kk * rows + m] : a[m * inner + kk];
                    areg[t] = v;
                }
                for (uint t = 0; t < BLOAD; ++t) {
                    uint idx = lid + t * THREADS;
                    uint kk = next + idx / BN;
                    uint n = tile_col + idx % BN;
                    float v = 0.0f;
                    if (n < columns && kk < k_stop)
                        v = trans_b ? b[n * inner + kk] : b[kk * columns + n];
                    breg[t] = v;
                }
            }

            for (uint kk = 0; kk < BK; kk += 8u) {
                simdgroup_matrix<_ET_, 8, 8> af[TM];
                simdgroup_matrix<_ET_, 8, 8> bf[TN];
                for (uint i = 0; i < TM; ++i)
                    simdgroup_load(af[i], As + (sg_m * TM * 8u + i * 8u) * BK + kk, BK);
                for (uint j = 0; j < TN; ++j)
                    simdgroup_load(bf[j], Bs + kk * BN + sg_n * TN * 8u + j * 8u, BN);
                for (uint i = 0; i < TM; ++i)
                    for (uint j = 0; j < TN; ++j)
                        simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
            }
        }
    }

    device float* dest = output + slice * rows * columns;

    /* A tile that lies wholly inside the matrix and has no epilogue to apply
     * can go straight from the accumulators to memory. The path below it exists
     * for the bias and the activation, which are per element and so have to see
     * the values one at a time; routing every tile through it costs two
     * barriers and a threadgroup round trip per row of fragments, and on a
     * short inner dimension that is a large share of the whole kernel. */
    const bool whole = tile_row + BM <= rows && tile_col + BN <= columns;
    if (whole && (epilogue == 0u || splits != 1u)) {
        for (uint i = 0; i < TM; ++i) {
            const uint r = tile_row + sg_m * TM * 8u + i * 8u;
            for (uint j = 0; j < TN; ++j) {
                const uint c = tile_col + sg_n * TN * 8u + j * 8u;
                simdgroup_store(acc[i][j], dest + r * columns + c, columns);
            }
        }
        return;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* scratch = pool + sgitg * 8u * SGW;
    const uint lane = lid % 32u;
    for (uint i = 0; i < TM; ++i) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < TN; ++j)
            simdgroup_store(acc[i][j], scratch + j * 8u, SGW);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 8u * SGW; e += 32u) {
            uint r = tile_row + sg_m * TM * 8u + i * 8u + e / SGW;
            uint c = tile_col + sg_n * SGW + e % SGW;
            if (r < rows && c < columns) {
                float total = scratch[e];
                if (splits == 1u) {
                    if (epilogue == 6u) total += bias[r * columns + c];
                    else if (epilogue != 0u) total += bias[c];
                    if (epilogue == 2u) total = max(total, 0.0f);
                    else if (epilogue == 3u) total = 1.0f / (1.0f + exp(-total));
                    else if (epilogue == 4u) total = tanh(total);
                    else if (epilogue == 5u) total = 0.5f * total * (1.0f + tanh(0.7978845608f * (total + 0.044715f * total * total * total)));
                }
                dest[r * columns + c] = total;
            }
        }
    }
}
"""

let DIRECT_LOOP = """
        for (uint k = k_begin; k < k_stop; k += 8u) {
            simdgroup_matrix<float, 8, 8> af[TM];
            simdgroup_matrix<float, 8, 8> bf[TN];
            for (uint i = 0; i < TM; ++i) { _LOAD_A_; }
            for (uint j = 0; j < TN; ++j) { _LOAD_B_; }
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < TN; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
"""

let LOAD_A_PLAIN = "simdgroup_load(af[i], a + (row0 + i * 8u) * inner + k, inner)"
let LOAD_A_TRANS = "simdgroup_load(af[i], a + k * rows + row0 + i * 8u, rows, ulong2(0), true)"
let LOAD_B_PLAIN = "simdgroup_load(bf[j], b + k * columns + col0 + j * 8u, columns)"
let LOAD_B_TRANS = "simdgroup_load(bf[j], b + (col0 + j * 8u) * inner + k, inner, ulong2(0), true)"

fn direct_loop(load_a: str, load_b: str) -> str {
    var text = DIRECT_LOOP.replace("_LOAD_A_", load_a)
    return text.replace("_LOAD_B_", load_b)
}

#: The four transpose combinations, each with its own copy of the inner loop.
#:
#: `simdgroup_load`'s transpose argument has to be a compile-time constant for
#: the shader compiler to pick an addressing mode; deciding it per fragment from
#: a `constant uint&` cost a factor of five on every shape measured.
fn direct_loops() -> str {
    var text = "    if (trans_a == 0u && trans_b == 0u) {\n"
    text = text + direct_loop(LOAD_A_PLAIN, LOAD_B_PLAIN)
    text = text + "    } else if (trans_a != 0u && trans_b == 0u) {\n"
    text = text + direct_loop(LOAD_A_TRANS, LOAD_B_PLAIN)
    text = text + "    } else if (trans_a == 0u && trans_b != 0u) {\n"
    text = text + direct_loop(LOAD_A_PLAIN, LOAD_B_TRANS)
    text = text + "    } else {\n"
    text = text + direct_loop(LOAD_A_TRANS, LOAD_B_TRANS)
    return text + "    }\n"
}

let DIRECT_TEMPLATE = """
kernel void NAME(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* output [[buffer(2)]],
                 device const float* bias [[buffer(3)]],
                 constant uint& rows [[buffer(4)]],
                 constant uint& inner [[buffer(5)]],
                 constant uint& columns [[buffer(6)]],
                 constant uint& trans_a [[buffer(7)]],
                 constant uint& trans_b [[buffer(8)]],
                 constant uint& epilogue [[buffer(9)]],
                 constant uint& splits [[buffer(10)]],
                 constant uint& stride_a [[buffer(11)]],
                 constant uint& stride_b [[buffer(12)]],
                 constant uint& stride_out [[buffer(13)]],
                 uint lid [[thread_index_in_threadgroup]],
                 uint wid [[threadgroup_position_in_grid]],
                 uint sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = _BM_;
    constexpr uint BN = _BN_;
    constexpr uint WM = _WM_;
    constexpr uint WN = _WN_;
    constexpr uint TM = BM / WM / 8u;
    constexpr uint TN = BN / WN / 8u;
    constexpr uint SGW = BN / WN;

    threadgroup float pool[WM * WN * 8u * SGW];

    const uint groups_x = columns / BN;
    const uint tiles = groups_x * (rows / BM);
    const uint per_item = tiles * splits;
    const uint item = wid / per_item;
    const uint local = wid - item * per_item;
    const uint slice = local / tiles;
    const uint tile = local - slice * tiles;
    a += item * stride_a;
    b += item * stride_b;
    output += item * stride_out;
    const uint sg_m = sgitg / WN;
    const uint sg_n = sgitg % WN;
    const uint row0 = (tile / groups_x) * BM + sg_m * TM * 8u;
    const uint col0 = (tile % groups_x) * BN + sg_n * TN * 8u;

    const uint chunk = ((inner + splits - 1u) / splits + 7u) / 8u * 8u;
    const uint k_begin = slice * chunk;
    const uint k_stop = min(inner, k_begin + chunk);

    simdgroup_matrix<float, 8, 8> acc[TM][TN];
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

_KLOOPS_
    device float* dest = output + slice * rows * columns;

    /* Straight from the accumulators when there is no epilogue to apply, for
     * the same reason the staged kernel does it: the threadgroup round trip
     * below exists only so that a bias and an activation can see one value at
     * a time. */
    if (epilogue == 0u || splits != 1u) {
        for (uint i = 0; i < TM; ++i)
            for (uint j = 0; j < TN; ++j)
                simdgroup_store(acc[i][j], dest + (row0 + i * 8u) * columns + col0 + j * 8u, columns);
        return;
    }

    threadgroup float* scratch = pool + sgitg * 8u * SGW;
    const uint lane = lid % 32u;
    for (uint i = 0; i < TM; ++i) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < TN; ++j)
            simdgroup_store(acc[i][j], scratch + j * 8u, SGW);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 8u * SGW; e += 32u) {
            uint r = row0 + i * 8u + e / SGW;
            uint c = col0 + e % SGW;
            float total = scratch[e];
            if (splits == 1u) {
                if (epilogue == 6u) total += bias[r * columns + c];
                else if (epilogue != 0u) total += bias[c];
                if (epilogue == 2u) total = max(total, 0.0f);
                else if (epilogue == 3u) total = 1.0f / (1.0f + exp(-total));
                else if (epilogue == 4u) total = tanh(total);
                else if (epilogue == 5u) total = 0.5f * total * (1.0f + tanh(0.7978845608f * (total + 0.044715f * total * total * total)));
            }
            dest[r * columns + c] = total;
        }
    }
}
"""

let REDUCE_SOURCE = """
kernel void split_reduce(device const float* partials [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         device const float* bias [[buffer(2)]],
                         constant uint& count [[buffer(3)]],
                         constant uint& columns [[buffer(4)]],
                         constant uint& splits [[buffer(5)]],
                         constant uint& epilogue [[buffer(6)]],
                         constant uint& batch [[buffer(7)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count * batch) return;
    const uint item = id / count;
    const uint local = id - item * count;
    float total = 0.0f;
    for (uint s = 0; s < splits; ++s) total += partials[(item * splits + s) * count + local];
    if (epilogue == 6u) total += bias[local];
    else if (epilogue != 0u) total += bias[local % columns];
    if (epilogue == 2u) total = max(total, 0.0f);
    else if (epilogue == 3u) total = 1.0f / (1.0f + exp(-total));
    else if (epilogue == 4u) total = tanh(total);
    else if (epilogue == 5u) total = 0.5f * total * (1.0f + tanh(0.7978845608f * (total + 0.044715f * total * total * total)));
    output[id] = total;
}

/* A row-major matrix copied into a wider and/or taller one, zero-filled.
 * MPS reads a matrix by its row bytes, which must be a multiple of sixteen;
 * a padded copy is what lets a 50-wide operand go there at all. The fill has
 * to be zero and not merely finite-garbage: padded inner positions are
 * multiplied by the other operand's padding, and 0 * NaN is NaN. */
kernel void gemm_pad(device const float* src [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                     constant uint& src_rows [[buffer(2)]],
                     constant uint& src_columns [[buffer(3)]],
                     constant uint& dst_rows [[buffer(4)]],
                     constant uint& dst_columns [[buffer(5)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= dst_rows * dst_columns) return;
    const uint r = id / dst_columns;
    const uint c = id - r * dst_columns;
    dst[id] = (r < src_rows && c < src_columns) ? src[r * src_columns + c] : 0.0f;
}

/* The padded product's top-left block copied out, with the epilogue applied
 * on the way, so a padded MPS product costs one pass over its result and not
 * two. */
kernel void gemm_unpad(device const float* src [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       device const float* bias [[buffer(2)]],
                       constant uint& rows [[buffer(3)]],
                       constant uint& columns [[buffer(4)]],
                       constant uint& src_columns [[buffer(5)]],
                       constant uint& epilogue [[buffer(6)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= rows * columns) return;
    const uint r = id / columns;
    const uint c = id - r * columns;
    float total = src[r * src_columns + c];
    if (epilogue == 6u) total += bias[id];
    else if (epilogue != 0u) total += bias[c];
    if (epilogue == 2u) total = max(total, 0.0f);
    else if (epilogue == 3u) total = 1.0f / (1.0f + exp(-total));
    else if (epilogue == 4u) total = tanh(total);
    else if (epilogue == 5u) total = 0.5f * total * (1.0f + tanh(0.7978845608f * (total + 0.044715f * total * total * total)));
    output[id] = total;
}
"""

#: One tile shape the GEMM kernel is instantiated for.
#:
#: `staged` kernels copy both operands through threadgroup memory, so they
#: convert to float16 on the way in and handle any shape; they carry four
#: fragments each way, so a threadgroup holds sixteen accumulators. `direct`
#: kernels read simdgroup fragments straight out of device memory: no barriers
#: and no conversion, which wins once a shape is large enough to be compute
#: bound, at the cost of requiring the tile to divide the matrix exactly. They
#: hold two fragments down and four across — device loads have far longer
#: latency than threadgroup loads, and sixteen accumulators alongside them
#: spilled, costing a factor of ten.
pub class Config {
    #: Kernel entry point name.
    pub let entry: str
    #: Whether operands are staged in threadgroup memory.
    pub let staged: bool
    #: Rows of the output tile a threadgroup owns.
    pub let tile_rows: int
    #: Columns of the output tile a threadgroup owns.
    pub let tile_columns: int
    #: Inner-dimension step consumed per iteration.
    pub let step: int
    #: Simdgroups down the tile.
    pub let warps_rows: int
    #: Simdgroups across the tile.
    pub let warps_columns: int

    pub fn init(
        self,
        entry: str,
        staged: bool,
        tile_rows: int,
        tile_columns: int,
        step: int,
        warps_rows: int,
        warps_columns: int
    ) {
        self.entry = entry
        self.staged = staged
        self.tile_rows = tile_rows
        self.tile_columns = tile_columns
        self.step = step
        self.warps_rows = warps_rows
        self.warps_columns = warps_columns
    }

    #: Threads in a threadgroup.
    pub fn group_size(self) -> int { return self.warps_rows * self.warps_columns * 32 }
}

let CONFIGS = [
    Config("gemm128x128", true, 128, 128, 16, 4, 4),
    Config("gemm128x64", true, 128, 64, 16, 4, 2),
    Config("gemm64x64", true, 64, 64, 16, 2, 2),
    Config("gemm64x32", true, 64, 32, 16, 2, 1),
    Config("gemm32x32", true, 32, 32, 16, 1, 1),
    #: The same tiles again over a deeper step. A staged kernel pays two
    #: barriers per step whatever the step is, so doubling `BK` halves what
    #: the barriers cost — at the price of twice the threadgroup memory, which
    #: is what keeps this from simply replacing the family above it.
    Config("gemm128x128k32", true, 128, 128, 32, 4, 4),
    Config("direct128x64", false, 128, 64, 8, 8, 2),
    Config("direct64x128", false, 64, 128, 8, 4, 4),
    Config("direct64x64", false, 64, 64, 8, 4, 2),
    Config("direct32x64", false, 32, 64, 8, 2, 2),
    Config("direct32x32", false, 32, 32, 8, 2, 1),
]

let SPLIT_CANDIDATES = [1, 2, 4, 8, 16]

#: Splits past sixteen, for a product whose output is so small that sixteen
#: slices of it still leave the device mostly idle.
#:
#: A tall-skinny product -- a gram `X^T X` over 100k samples of 50 features, or
#: `X^T y` -- has four or two output tiles and a hundred thousand steps of K.
#: Sixteen splits was the ceiling, and the tuner sat on it: 3.7 ms for X^T X
#: where 64 splits gives 1.5, and 3.5 ms for X^T y where 256 gives 1.1. The
#: best count differs between the two, so these are candidates and not a rule.
#: Offered only when sixteen splits of the configuration's tiles stay under
#: `TARGET_GROUPS`; anywhere else they would only multiply the partials.
let DEEP_SPLITS = [32, 64, 128, 256]

#: Fewest output tiles a configuration may produce before a smaller tile wins.
let MIN_TILES = 16

#: Threadgroups a split is allowed to grow the dispatch to.
let TARGET_GROUPS = 256

#: Products smaller than this take the static plan; measuring costs more than
#: the whole multiplication and the timings would be noise anyway.
let TUNE_MIN_WORK = 4194304

#: Multiply-adds above which the static plan is MPSGraph. Past about this size
#: the graph's thirty-odd microseconds of host encode are lost in the product,
#: and every tuned plan cache on record chose it from here up.
let MPS_STATIC_WORK = 268435456

#: Fewest steps of K a product needs before `deep_product` will call it deep.
let DEEP_MIN_INNER = 16384

#: Fewest steps of K each split of a static plan keeps; see `filling_splits`.
let STATIC_SPLIT_DEPTH = 256

#: The same floor for a staged static plan. Staged kernels run where no direct
#: tile divides the shape, and there the split is what fills the device: 300
#: cubed ran at 270 GFLOPS unsplit on `gemm64x32` and 625 split four ways,
#: 75 steps of K each.
let STAGED_SPLIT_DEPTH = 64

#: Multiply-adds per element the padded copies move, above which a large
#: product that MPS can only read through zero-padded copies still goes to the
#: graph statically. A copy of the big operand is a full pass over it, and on a
#: tall-skinny output that is most of what the product costs: 500000x130x10
#: (9 multiply-adds per copied element) ran 29.5 ms padded on the graph against
#: 24.9 on `gemm64x32`, where 100000x50x128 (123) ran 0.97 padded against 1.5.
let PAD_MPS_RATIO = 32

#: How many times a bucket must be asked for before it may be tuned. A shape
#: that is used once or twice can never earn back what measuring it costs.
let TUNE_MIN_REQUESTS = 3

#: Estimated seconds of GEMM work one shape must have run, on its static plan,
#: before its bucket is tuned.
#:
#: Tuning every new shape on its first call stalled whole programs: a 128-token
#: generation tuned 385 row counts at half a second each, 204 s against 0.93 s
#: warm, and a first Ridge fit spent 8-19 s tuning a product it then ran in a
#: few milliseconds. This is the ski-rental rule -- keep paying the static
#: plan's price until what has been paid is comparable to what tuning costs,
#: then tune -- so a shape is never tuned before it has cost a meaningful
#: fraction of a tuning pass, and a program that is short never pays one at
#: all. The work is estimated from the shape, never timed: timing it would
#: need a wait on every call.
let TUNE_PAYBACK_SECONDS = 0.15

#: The rate and per-call cost that estimate assumes. Deliberately slower than
#: the device usually runs, so a heavy shape crosses the line within the first
#: few hundred calls of a warm-up rather than in the middle of a timed loop.
let TUNE_ASSUMED_FLOPS = 2.0e12
let TUNE_ASSUMED_CALL_SECONDS = 0.00002

#: And the rate it assumes operands move at, for a product that is bound by
#: memory rather than arithmetic: a 100000x50x1 product does a millionth of
#: the arithmetic of a 1024-cubed one and still reads 20 MB.
let TUNE_ASSUMED_BYTES_PER_SECOND = 2.0e11

#: The arithmetic rate assumed instead where the static plan is a staged
#: kernel, which is where the static plan is furthest behind a tuned one: the
#: staged kernels ran ragged shapes at 0.1-0.9 TFLOPS, where the graph ran
#: 100x4096x300 at 1.7 and 2000x20x2000 at 1.1. Charged at 2 TFLOPS, those
#: buckets needed over a thousand calls -- half a second of real work, more
#: than a short program does -- before they were tuned at all.
let TUNE_ASSUMED_STAGED_FLOPS = 5.0e11

#: Tuning passes a bucket may lose to a queue that would not drain in time
#: (see `tune_bucket`) before it settles on its static plan for the process.
let TUNE_MAX_ATTEMPTS = 3

#: Wall-clock ceiling on one tuning pass, warm-up included. Candidates still
#: unmeasured when it runs out are not measured.
let TUNE_BUDGET_SECONDS = 0.3

#: A candidate whose best reading is this many times the best so far is not
#: measured again.
let TUNE_ABANDON = 2.0

#: A kernel plan replaces MPSGraph only when it is at least this much faster.
#: A plan is measured once and believed until the epoch moves, and in-tune
#: readings of the graph have run high against every sustained measurement:
#: `direct32x32:1` was cached for 1024x768x768 at 0.29 ms against 0.21 for the
#: graph. The margin keeps a near-tie on the side that is sustained. The
#: graph's two precisions are not held to it against each other: on the
#: 784-deep bench products half precision is a steady 6-14% ahead, and a
#: margin threw that away.
let TUNE_MPS_MARGIN = 1.1

#: A sample is timed over enough repeats to run for at least this long, so
#: dispatch overhead cannot decide the winner for a short kernel.
let TUNE_MIN_SECONDS = 0.004

#: How many rounds the candidates are timed in, turn about; each keeps its
#: fastest reading. Up to `TUNE_ROUNDS` run while the budget lasts -- a long
#: candidate list can be cut short inside one, and then the winner is chosen
#: among candidates holding unequal numbers of readings -- and more, up to
#: `TUNE_MAX_ROUNDS`, only while a whole round still fits, which for a large
#: shape down to two graph candidates is several more readings of a decision
#: that is otherwise inside the device's drift.
let TUNE_ROUNDS = 3
let TUNE_MAX_ROUNDS = 8

#: What a synchronised sample costs beyond its kernels, for budgeting rounds.
let TUNE_SYNC_SECONDS = 0.0003

#: Ceiling on those repeats, so tuning a slow candidate stays bounded.
let TUNE_MAX_REPEATS = 128

#: Dispatches fired before any candidate is timed, and the longest that may
#: take.
#:
#: The device reaches its working clock after a few dozen dispatches, and the
#: candidates are timed one after another -- so without this the first one is
#: read on a cold clock and the rest on a warm one. That is a bias and not
#: noise: `mpsgraph` is always first in the list, and one cold reading of it
#: wrote `mpsgraph16` into the plan cache for 4096x784x8192, where it runs at
#: 5400 GFLOPS against float32's 9200. A cached plan is never re-measured, so
#: that stood until the epoch moved.
let TUNE_WARMUP = 40
let TUNE_WARM_SECONDS = 0.05
let TUNE_WARM_STRIDE = 8

fn instantiate(config: Config, element: str) -> str {
    let template = if config.staged { TEMPLATE } else { DIRECT_TEMPLATE }
    var text = template.replace("NAME", config.entry)
    if not config.staged { text = text.replace("_KLOOPS_", direct_loops()) }
    text = text.replace("_ET_", element)
    text = text.replace("_BM_", f"{config.tile_rows}")
    text = text.replace("_BN_", f"{config.tile_columns}")
    text = text.replace("_BK_", f"{config.step}")
    text = text.replace("_WM_", f"{config.warps_rows}")
    text = text.replace("_WN_", f"{config.warps_columns}")
    return text
}

fn build_source(element: str) -> str {
    var text = "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n"
    for config in CONFIGS { text = text + instantiate(config, element) }
    return text + REDUCE_SOURCE
}

var _sources: dict[str, str] = {}

fn source_for(element: str) -> str {
    let existing = _sources.get(element, null)
    if existing is not null { return existing }
    let built = build_source(element)
    _sources[element] = built
    return built
}

var _kernels: dict[str, Kernel] = {}

fn kernel_for(element: str, entry: str) -> Kernel {
    let key = element + "/" + entry
    let existing = _kernels.get(key, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(source_for(element), entry)
    _kernels[key] = kernel
    return kernel
}

fn element_type() -> str {
    return if gpu.mixed_precision() { "half" } else { "float" }
}

var _forced_read = false
var _forced: Plan? = null

fn forced_plan() -> Plan? {
    if _forced_read { return _forced }
    let text = env("JAITENSOR_GEMM_PLAN")
    if text is not null and text.len() > 0 {
        let parsed = plan_from_text(text)
        #: It used to fall back to tuning without a word, so `mpsgraph` --
        #: which needed to be spelled `mpsgraph:1` -- measured nothing it
        #: claimed to and looked like a result. The value is not marked read
        #: until it parses, so every product raises, not only the first: a
        #: harness that catches per test would otherwise run all but one of
        #: them unforced.
        if parsed is null {
            throw ValueError(
                f"JAITENSOR_GEMM_PLAN={text} names no plan; expected a kernel name or mpsgraph, optionally followed by :splits"
            )
        }
        _forced = parsed
    }
    _forced_read = true
    return _forced
}

var _autotune = -1

fn autotune_enabled() -> bool {
    if _autotune < 0 { _autotune = if (env("JAITENSOR_AUTOTUNE") ?? "1") != "0" { 1 } else { 0 } }
    return _autotune == 1
}

var _verbose = -1

fn autotune_verbose() -> bool {
    if _verbose < 0 {
        _verbose = if (env("JAITENSOR_AUTOTUNE_VERBOSE") ?? "0") != "0" { 1 } else { 0 }
    }
    return _verbose == 1
}

#: How one shape is routed: a tile configuration and a K split, or MPSGraph.
pub class Plan {
    #: `"mpsgraph"` or the kernel entry point that runs the product.
    pub let backend: str
    #: Index into the built-in tile configurations, or -1 for MPSGraph.
    pub let config: int
    #: Number of independent K ranges the product is broken into.
    pub let splits: int

    pub fn init(self, backend: str, config: int, splits: int) {
        self.backend = backend
        self.config = config
        self.splits = splits
    }

    #: Stable text form, also what the on-disk cache stores.
    pub fn to_str(self) -> str { return f"{self.backend}:{self.splits}" }
}

let MPS_PLAN = Plan("mpsgraph", -1, 1)

#: `MPSMatrixMultiplication` was tried alongside this and taken out again.
#:
#: It is the same arithmetic on the same device and lands within a few percent
#: either way -- 6% ahead at 1024x3072x768, 7% behind at 512x8192x8192 -- so as
#: a fourth candidate it should have been free money on the shapes it wins.
#: It was not: adding it took the whole suite from 1.81x to 1.57x and vit-block
#: from 0.96x to 0.49x, on shapes where the tuner still chose the graph. One
#: more candidate is one more plan to time for every shape a model touches, and
#: the tuning costs more than the few percent is worth. A route has to earn
#: back what searching for it costs.

#: The same graph asked to run in half precision.
#:
#: Whether the conversion pays depends on the shape rather than on any rule
#: worth writing down: it costs a pass over A, over B, and twice over the
#: result, and it saves on the multiply. Both are timed and the winner kept.
let MPS16_PLAN = Plan("mpsgraph16", -2, 1)

#: Whether every matrix MPS would read has rows a multiple of sixteen bytes:
#: `a` and `b` as stored, and the result.
fn mps_aligned(rows: int, inner: int, columns: int, trans_a: bool, trans_b: bool) -> bool {
    let a_columns = if trans_a { rows } else { inner }
    let b_columns = if trans_b { inner } else { columns }
    return a_columns % 4 == 0 and b_columns % 4 == 0 and columns % 4 == 0
}

#: Whether the MPSGraph route can run this shape: natively when it is aligned,
#: and through a zero-padded copy otherwise (see `encode_mps`).
#:
#: Never for an empty inner axis: MPS refuses `k == 0` outright, and the
#: product is all zeros, which any kernel writes.
fn mps_possible(rows: int, inner: int, columns: int, trans_a: bool, trans_b: bool) -> bool {
    if inner <= 0 or not gpu.is_available() { return false }
    return _PAD or mps_aligned(rows, inner, columns, trans_a, trans_b)
}

#: Elements the padded MPS route copies for this shape: each operand that has
#: to be padded, and the padded result when it has to be cut back. Zero when
#: MPS reads the shape as stored. Mirrors `encode_mps`.
fn padded_elements(rows: int, inner: int, columns: int, trans_a: bool, trans_b: bool) -> int {
    if mps_aligned(rows, inner, columns, trans_a, trans_b) { return 0 }
    let padded_rows = if trans_a { round_up_4(rows) } else { rows }
    let padded_inner = if not trans_a or trans_b { round_up_4(inner) } else { inner }
    let padded_columns = round_up_4(columns)
    var copied = 0
    if padded_rows != rows or padded_inner != inner { copied += padded_rows * padded_inner }
    if padded_inner != inner or padded_columns != columns {
        copied += padded_inner * padded_columns
    }
    if padded_rows != rows or padded_columns != columns { copied += padded_rows * padded_columns }
    return copied
}

#: Whether the graph's own speed pays for the copies padding it would take.
fn padding_pays(rows: int, inner: int, columns: int, trans_a: bool, trans_b: bool) -> bool {
    return rows * inner * columns >=
        PAD_MPS_RATIO * padded_elements(rows, inner, columns, trans_a, trans_b)
}

fn tile_count(config: Config, rows: int, columns: int) -> int {
    let across = (columns + config.tile_columns - 1) // config.tile_columns
    let down = (rows + config.tile_rows - 1) // config.tile_rows
    return across * down
}

#: Every split count the planner may consider, in increasing order.
fn split_counts() -> list[int] {
    if not _DEEP_SPLITS { return SPLIT_CANDIDATES }
    return SPLIT_CANDIDATES + DEEP_SPLITS
}

fn split_allowed(config: Config, rows: int, inner: int, columns: int, splits: int) -> bool {
    if splits == 1 { return true }
    if inner // splits < 64 { return false }
    let tiles = tile_count(config, rows, columns)
    if splits > 16 and tiles * 16 >= TARGET_GROUPS { return false }
    return tiles * splits <= 8192
}

fn config_allowed(config: Config, rows: int, inner: int, columns: int) -> bool {
    if config.staged { return true }
    return rows % config.tile_rows == 0 and columns % config.tile_columns == 0 and inner % 8 == 0
}

fn config_index(entry: str) -> int {
    for (index, config) in CONFIGS.enumerate() {
        if config.entry == entry { return index }
    }
    throw ValueError(f"no GEMM configuration named {entry}")
}

let DIRECT_NARROW = config_index("direct32x32")
let DIRECT_WIDE = config_index("direct32x64")

#: The most splits of `config` that keep the dispatch within `TARGET_GROUPS`
#: and leave each split at least `STATIC_SPLIT_DEPTH` steps of K.
#:
#: The depth floor is for the static plan only (the tuner may go shallower
#: where it measures a gain): a split costs a second dispatch to sum the
#: partials, and on a short K that is most of the product -- 128 cubed ran
#: 16.3 us split in two against 12.6 us whole.
fn filling_splits(config: Config, rows: int, inner: int, columns: int) -> int {
    var depth = 0
    if _STATIC_DIRECT {
        depth = if config.staged { STAGED_SPLIT_DEPTH } else { STATIC_SPLIT_DEPTH }
    }
    var splits = 1
    for candidate in split_counts() {
        if not split_allowed(config, rows, inner, columns, candidate) { continue }
        if candidate > 1 and inner // candidate < depth { continue }
        if tile_count(config, rows, columns) * candidate <= TARGET_GROUPS { splits = candidate }
    }
    return splits
}

#: Whether a product is far deeper than it is wide: an `X^T X` or `X^T y` over
#: many samples. Where no direct kernel divides one, MPS splits K itself and
#: beat every staged plan -- 0.80 ms for a 100k x 50 gram against 1.45 for the
#: best split kernel, and 0.46 against 0.82 for `X^T y` -- once its encode is
#: small beside the work. Where a direct kernel does divide it, the graph is
#: no rule: at widths 96 and 160 it ran a 40000-deep gram four times slower
#: than `direct32x32:16`, and at 80, 112 and 192 it tied.
fn deep_product(rows: int, inner: int, columns: int) -> bool {
    return inner >= DEEP_MIN_INNER and inner >= 16 * rows and inner >= 16 * columns
}

#: Whether the output has too few tiles to fill the device without splitting
#: K, even in the narrowest direct tile. These are the products where a split
#: kernel and the graph trade places shape by shape, so neither may be assumed.
fn small_output(rows: int, columns: int) -> bool {
    return tile_count(CONFIGS[DIRECT_NARROW], rows, columns) < TARGET_GROUPS
}

#: The direct kernel the static plan would use for this shape, or -1.
fn static_direct(rows: int, inner: int, columns: int) -> int {
    let wide = CONFIGS[DIRECT_WIDE]
    if config_allowed(wide, rows, inner, columns) and
        tile_count(wide, rows, columns) >= TARGET_GROUPS {
        return DIRECT_WIDE
    }
    if config_allowed(CONFIGS[DIRECT_NARROW], rows, inner, columns) { return DIRECT_NARROW }
    return -1
}

#: The staged tile a static plan uses where no direct kernel fits.
#:
#: `gemm64x32` wherever it still makes `MIN_TILES` threadgroups, else
#: `gemm32x32`. The largest tile was the rule before, and on a narrow output it
#: computed mostly padding: 100000x50x1 ran at 6 GFLOPS on `gemm128x128` and
#: 20 on `gemm64x32`. Across the ragged shapes a tuner was run on --
#: 300x300x300, 1000x1000x96, 100x4096x300, 2000x20x2000, 500000x130x10 --
#: `gemm64x32`, split to fill the device, was the best staged plan or within
#: a few percent of it every time.
fn static_staged(rows: int, columns: int) -> int {
    let medium = config_index("gemm64x32")
    if tile_count(CONFIGS[medium], rows, columns) >= MIN_TILES { return medium }
    return config_index("gemm32x32")
}

#: The routing a shape gets without measuring anything.
#:
#: A direct kernel runs wherever its tile divides the shape: staged
#: `gemm32x32` cost 15.6 us of GPU time per 64-cubed product against about 5
#: for `direct32x32`, and among tuned plans the direct kernels won 59 times to
#: the staged one's once. The graph takes a large product whose output fills
#: the device -- where every tuned cache on record chose it -- if its padded
#: copies, when it needs them, are small beside the product; and a deep one
#: no direct kernel divides (see `deep_product`). An output too small to fill
#: the device keeps the direct kernel and splits K instead: there the best
#: plan for 256x5000x256 and 256x8192x128 was `direct32x32:16`, ahead of the
#: graph. Anything else ragged takes the staged tile `static_staged` picks.
#: Either way K is then split until the device has enough threadgroups to stay
#: busy; shrinking the tile instead costs arithmetic intensity, so splitting is
#: always tried first.
#:
#: Never the graph for a ragged product below `MPS_STATIC_WORK`, though the
#: graph won the tuning pass for most of them: a new MPSGraph shape costs about
#: 3.2 ms on its first call, against 0.4 for the kernel, and a program that
#: walks through row counts -- a generation, the folds of a cross-validation --
#: would pay that for every one of them. A bucket that is used enough is tuned
#: onto the graph anyway.
pub fn default_plan(
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false
) -> Plan {
    if _STATIC_DIRECT {
        let work = rows * inner * columns
        let direct = static_direct(rows, inner, columns)
        let small = small_output(rows, columns)
        if mps_possible(rows, inner, columns, trans_a, trans_b) and not (direct >= 0 and small) {
            let deep = work >= TUNE_MIN_WORK and deep_product(rows, inner, columns)
            if deep and direct < 0 { return MPS_PLAN }
            if work >= MPS_STATIC_WORK and padding_pays(rows, inner, columns, trans_a, trans_b) {
                return MPS_PLAN
            }
        }
        let chosen = if direct >= 0 { direct } else { static_staged(rows, columns) }
        let config = CONFIGS[chosen]
        return Plan(config.entry, chosen, filling_splits(config, rows, inner, columns))
    }
    var chosen = 4
    for (index, config) in CONFIGS.enumerate() {
        if config.staged and tile_count(config, rows, columns) >= MIN_TILES {
            chosen = index
            break
        }
    }
    let config = CONFIGS[chosen]
    return Plan(config.entry, chosen, filling_splits(config, rows, inner, columns))
}

#: Whether `plan` can run this exact shape.
fn plan_legal(
    plan: Plan,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool,
    trans_b: bool
) -> bool {
    if plan.config < 0 { return mps_possible(rows, inner, columns, trans_a, trans_b) }
    let config = CONFIGS[plan.config]
    if not config_allowed(config, rows, inner, columns) { return false }
    return split_allowed(config, rows, inner, columns, plan.splits)
}

var _cache: dict[str, Plan] = {}
var _cache_loaded = false
var _cache_dirty = false

fn cache_path() -> Path {
    let override = env("JAITENSOR_GEMM_CACHE")
    if override is not null { return Path(override) }
    return Path(home_dir()).join(".cache", "jaithon", "gemm-plans.tsv")
}

var _epochs: dict[str, str] = {}

#: What the cached plans were measured against, as a digest of the thing itself.
#:
#: A cached plan is a measurement of a set of candidates that no longer exists
#: once that set changes, and nothing would ever re-measure it: the entry is
#: found, believed, and used forever. Making this part of the key retires every
#: old entry the first time a changed build runs, which costs one tuning pass.
#:
#: This used to be a hand-bumped integer, and that is the failure it was meant
#: to prevent, one level up: edit a kernel, forget the bump, and every plan on
#: the machine is stale and never re-measured -- silently, since a stale plan
#: still computes the right answer, only slower. Derived from `source_for`, it
#: covers every kernel body and every tile parameter, because `instantiate`
#: substitutes those into the text. The three constants that steer the choice
#: without appearing in the source are appended by hand.
fn tune_epoch(element: str) -> str {
    let existing = _epochs.get(element, null)
    if existing is not null { return existing }
    #: The two are hashed separately and combined, rather than concatenated.
    #: Appending to the memoised source does not corrupt it -- a concatenation
    #: writes PAST the view it was handed, and every older view still ends
    #: where it did. What it does do is leave that older view no longer
    #: NUL-terminated, and `gpu_compile` used to hand `chars` to the Metal
    #: compiler as a C string: the next `Kernel.compile` read past the end of
    #: the source and failed on `,1,2,4,8,16|16|256|2`. Fixed in
    #: builtins_gpu.c, but hashing the two separately is both cheaper and
    #: leaves the memoised string alone, so it stays that way.
    #: `policy` names how plans are keyed and chosen. It moved when plans
    #: became per bucket and the tuner learned to alternate and abandon,
    #: and when the static plan learned the direct kernels -- 305 of the 417
    #: rows a machine had cached were staged static plans, which nothing
    #: would otherwise have re-derived. It moved again (4) when kernels were
    #: let back in against a static graph plan on small outputs and padded
    #: shapes, and when a starved pass stopped being written down. The
    #: switches are in it because each changes what a tuning pass can choose
    #: between.
    var extra = f"|{MIN_TILES}|{TARGET_GROUPS}|policy=lazy-buckets-4|{MPS_STATIC_WORK}|{TUNE_MPS_MARGIN}"
    extra = extra + f"|{PAD_MPS_RATIO}|{STAGED_SPLIT_DEPTH}"
    extra = extra + f"|{_LAZY_TUNE}|{_PAD}|{_STATIC_DIRECT}|{_DEEP_SPLITS}"
    for candidate in split_counts() { extra = extra + f",{candidate}" }
    #: FNV-1a over the bytes (jaiHashBytes), so it is the same number in every
    #: process on every machine -- which it has to be, for a file on disk.
    let digest = f"{(hash(source_for(element)) ^ hash(extra)) & 0x7FFFFFFFFFFF}"
    _epochs[element] = digest
    return digest
}

#: The extent a plan is keyed on: the next power of two up to 64, then the
#: next quarter octave (64, 80, 96, 112, 128, 160, ...). Zero is its own
#: bucket: an empty inner axis shared the one-deep bucket, and a graph plan
#: tuned there failed on it, because MPS refuses `k == 0`.
#:
#: A plan is a property of a shape's rough proportions far more than of its
#: exact extent, and keying on the exact one is what made every new row count
#: of a generation, and every new `n_samples` of a cross-validation fold, a
#: fresh tuning pass. A quarter octave keeps shapes within 25% of each other
#: together. A bucket is tuned on whichever of its members earns the pass (see
#: `plan_for`), and that member's plan is then borrowed by every other one it
#: is legal for; every bucket above 64 is a multiple of 16, so it usually is.
#: Where it is not -- a direct kernel whose tile does not divide a ragged
#: member -- `plan_legal` sends that member to the static plan instead.
pub fn bucket_extent(extent: int) -> int {
    if extent <= 0 { return 0 }
    if extent == 1 { return 1 }
    if extent <= 64 {
        var power = 1
        while power < extent { power = power * 2 }
        return power
    }
    var octave = 64
    while octave * 2 <= extent { octave = octave * 2 }
    let step = octave // 4
    return (extent + step - 1) // step * step
}

fn shape_key(rows: int, inner: int, columns: int, trans_a: bool, trans_b: bool) -> str {
    let a_flag = if trans_a { 1 } else { 0 }
    let b_flag = if trans_b { 1 } else { 0 }
    let element = element_type()
    let prefix = f"{gpu.device_name()}\tv{tune_epoch(element)}\t{element}"
    if not _LAZY_TUNE { return f"{prefix}\t{rows}\t{inner}\t{columns}\t{a_flag}\t{b_flag}" }
    return f"{prefix}\t{bucket_extent(rows)}\t{bucket_extent(inner)}\t{columns}\t{a_flag}\t{b_flag}"
}

#: The plan-cache row this shape reads and writes: the bucket it falls in,
#: under this build's epoch, on this device.
pub fn cache_key(
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false
) -> str {
    return shape_key(rows, inner, columns, trans_a, trans_b)
}

var _memo_key = -1
var _memo: Route? = null

#: A training step asks for the same handful of shapes on every batch, so the
#: plan lookup is memoised on the shape itself: a full lookup builds a cache key
#: string, and doing that per dispatch cost more than encoding the kernel. One
#: entry in front catches a loop over one shape; `_routes` behind it, keyed on
#: this number and checked against the exact shape, catches a step that
#: alternates between a dozen.
#:
#: Wrapping arithmetic throughout. A mixed hash of three dimensions masked to 48
#: bits, multiplied again, is past 2^63 for any `rows` past a few hundred
#: thousand -- and `conv-wide`'s backward pass asks for exactly that shape, so
#: with `*` the whole benchmark died on an OverflowError raised from inside the
#: plan lookup. Nothing about a hash wants trapping arithmetic. `&` binds looser
#: than `+%`, so each line masks its whole mix.
fn memo_key(rows: int, inner: int, columns: int, trans_a: bool, trans_b: bool) -> int {
    const MASK = 0xFFFFFFFFFFFF
    var key = rows *% 6291469 +% inner & MASK
    key = key *% 6291469 +% columns & MASK
    key = key * 2 + (if trans_a { 1 } else { 0 })
    return key * 2 + (if trans_b { 1 } else { 0 })
}

#: Read back what `Plan.to_str` wrote, or null when the text names no plan.
#:
#: A bare name means one split, so `mpsgraph` reads as `mpsgraph:1`. It used
#: to need the count spelled out and returned null without it, which for
#: `JAITENSOR_GEMM_PLAN` meant the forced plan silently was not forced.
pub fn plan_from_text(text: str) -> Plan? {
    let parts = text.split(":")
    if parts.len() < 1 or parts.len() > 2 { return null }
    var splits = 1
    if parts.len() == 2 {
        try {
            splits = int(parts[1])
        } catch _error: Error {
            return null
        }
        if splits < 1 { return null }
    }
    if parts[0] == "mpsgraph" { return MPS_PLAN }
    if parts[0] == "mpsgraph16" { return MPS16_PLAN }
    for (index, config) in CONFIGS.enumerate() {
        if config.entry == parts[0] { return Plan(config.entry, index, splits) }
    }
    return null
}

fn load_cache() -> void {
    if _cache_loaded { return }
    _cache_loaded = true
    let path = cache_path()
    if not path.exists() { return }
    for line in path.read_text().split("\n") {
        if line.len() == 0 { continue }
        let parts = line.rsplit("\t", 1)
        if parts.len() != 2 { continue }
        let plan = plan_from_text(parts[1])
        if plan is not null { _cache[parts[0]] = plan }
    }
}

fn store_cache() -> void {
    if not _cache_dirty { return }
    _cache_dirty = false
    let path = cache_path()
    var merged: dict[str, Plan] = {}
    if path.exists() {
        try {
            for line in path.read_text().split("\n") {
                if line.len() == 0 { continue }
                let parts = line.rsplit("\t", 1)
                if parts.len() != 2 { continue }
                let stored = plan_from_text(parts[1])
                if stored is not null { merged[parts[0]] = stored }
            }
        } catch _error: Error {
            merged = {}
        }
    }
    for (key, plan) in _cache.items() { merged[key] = plan }
    var text = ""
    for (key, plan) in merged.items() { text = text + key + "\t" + plan.to_str() + "\n" }
    try {
        path.parent().mkdir(parents: true, exist_ok: true)
        path.write_text(text)
    } catch _error: Error {
        return
    }
}

var _scratch: Buffer? = null
var _scratch_count = 0

fn scratch_for(count: int) -> Buffer {
    let existing = _scratch
    if existing is not null and _scratch_count >= count { return existing }
    if existing is not null { existing.free() }
    let grown = Buffer(count)
    _scratch = grown
    _scratch_count = count
    return grown
}

fn round_up_4(extent: int) -> int { return (extent + 3) // 4 * 4 }

#: `source`, a row-major `rows x columns` matrix, as a zero-filled
#: `padded_rows x padded_columns` one -- or `source` itself when that is
#: already its shape. The flag says a copy was made and is the caller's to free.
fn padded_operand(
    source: Buffer,
    rows: int,
    columns: int,
    padded_rows: int,
    padded_columns: int
) -> tuple[Buffer, bool] {
    if rows == padded_rows and columns == padded_columns { return (source, false) }
    let copy = Buffer(padded_rows * padded_columns)
    kernel_for(element_type(), "gemm_pad").dispatch_async(
        [source, copy],
        padded_rows * padded_columns,
        [rows, columns, padded_rows, padded_columns]
    )
    return (copy, true)
}

#: One product on MPSGraph, zero-padding whatever MPS cannot read as stored.
#:
#: MPS describes a matrix by its row bytes, so it takes only rows that are a
#: multiple of four floats; everything else used to fall to the tile kernels,
#: and a 100k x 50 gram ran 3.7 ms there against 0.9 at width 52. Here each
#: extent that some matrix uses as its row length is rounded up to four -- the
#: inner one too, when either operand stores it innermost -- the operands that
#: change shape are copied into zero-filled buffers, and the product of those
#: is cut back to `rows x columns`, with the epilogue applied on the way out.
#: Zeros along the inner axis add nothing, and the extra rows and columns of
#: the result are never read. The copies are released as soon as they are
#: encoded: the queue is serial, so whatever reuses them runs after this does.
#: `a` and `b` being one buffer read the same way -- `X^T X` -- pad once.
fn encode_mps(
    a: Buffer,
    b: Buffer,
    output: Buffer,
    bias: Buffer,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool,
    trans_b: bool,
    epilogue: int,
    half: bool
) -> void {
    let element = element_type()
    if mps_aligned(rows, inner, columns, trans_a, trans_b) {
        gpu.matmul_async(
            a,
            b,
            output,
            rows,
            inner,
            columns,
            trans_a: trans_a,
            trans_b: trans_b,
            half: half
        )
        if epilogue != 0 {
            kernel_for(element, "split_reduce").dispatch_async(
                [output, output, bias],
                rows * columns,
                [rows * columns, columns, 1, epilogue, 1]
            )
        }
        return
    }
    let padded_rows = if trans_a { round_up_4(rows) } else { rows }
    let padded_inner = if not trans_a or trans_b { round_up_4(inner) } else { inner }
    let padded_columns = round_up_4(columns)
    #: Stored shapes, before and after: `a` is `[inner, rows]` when it is read
    #: transposed and `[rows, inner]` otherwise; `b` is `[columns, inner]` or
    #: `[inner, columns]`.
    let a_rows = if trans_a { inner } else { rows }
    let a_columns = if trans_a { rows } else { inner }
    let a_padded_rows = if trans_a { padded_inner } else { padded_rows }
    let a_padded_columns = if trans_a { padded_rows } else { padded_inner }
    let b_rows = if trans_b { columns } else { inner }
    let b_columns = if trans_b { inner } else { columns }
    let b_padded_rows = if trans_b { padded_columns } else { padded_inner }
    let b_padded_columns = if trans_b { padded_inner } else { padded_columns }
    let (left, owns_left) = padded_operand(a, a_rows, a_columns, a_padded_rows, a_padded_columns)
    let shared = a.handle_of() == b.handle_of() and
        a_rows == b_rows and
        a_columns == b_columns and
        a_padded_rows == b_padded_rows and
        a_padded_columns == b_padded_columns
    let (right, owns_right) = if shared {
        (left, false)
    } else {
        padded_operand(b, b_rows, b_columns, b_padded_rows, b_padded_columns)
    }
    if padded_rows == rows and padded_columns == columns {
        gpu.matmul_async(
            left,
            right,
            output,
            padded_rows,
            padded_inner,
            padded_columns,
            trans_a: trans_a,
            trans_b: trans_b,
            half: half
        )
        if epilogue != 0 {
            kernel_for(element, "split_reduce").dispatch_async(
                [output, output, bias],
                rows * columns,
                [rows * columns, columns, 1, epilogue, 1]
            )
        }
    } else {
        let wide = Buffer(padded_rows * padded_columns)
        gpu.matmul_async(
            left,
            right,
            wide,
            padded_rows,
            padded_inner,
            padded_columns,
            trans_a: trans_a,
            trans_b: trans_b,
            half: half
        )
        kernel_for(element, "gemm_unpad").dispatch_async(
            [wide, output, bias],
            rows * columns,
            [rows, columns, padded_columns, epilogue]
        )
        wide.free()
    }
    if owns_left { left.free() }
    if owns_right { right.free() }
}

fn encode(
    plan: Plan,
    a: Buffer,
    b: Buffer,
    output: Buffer,
    bias: Buffer,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool,
    trans_b: bool,
    epilogue: int,
    batch: int = 1,
    stride_a: int = 0,
    stride_b: int = 0,
    stride_out: int = 0
) -> void {
    if plan.config < 0 {
        for item in 0..batch {
            let a_part = if item == 0 { a } else { a.view(item * stride_a, stride_a) }
            let b_part = if item == 0 { b } else { b.view(item * stride_b, stride_b) }
            let out_part = if item == 0 {
                output
            } else {
                output.view(item * stride_out, stride_out)
            }
            encode_mps(
                a_part,
                b_part,
                out_part,
                bias,
                rows,
                inner,
                columns,
                trans_a,
                trans_b,
                epilogue,
                plan.config == -2
            )
            if item != 0 {
                a_part.free()
                b_part.free()
                out_part.free()
            }
        }
        return
    }
    let config = CONFIGS[plan.config]
    let element = element_type()
    let kernel = kernel_for(element, config.entry)
    let a_flag = if trans_a { 1 } else { 0 }
    let b_flag = if trans_b { 1 } else { 0 }
    let width = config.group_size()
    let groups = tile_count(config, rows, columns) * plan.splits * batch
    if plan.splits == 1 {
        kernel.dispatch_async(
            [a, b, output, bias],
            groups * width,
            [rows, inner, columns, a_flag, b_flag, epilogue, 1, stride_a, stride_b, stride_out],
            group_size: width
        )
        return
    }
    let per_item = rows * columns * plan.splits
    let partials = scratch_for(per_item * batch)
    kernel.dispatch_async(
        [a, b, partials, bias],
        groups * width,
        [rows, inner, columns, a_flag, b_flag, epilogue, plan.splits, stride_a, stride_b, per_item],
        group_size: width
    )
    let reduce = kernel_for(element, "split_reduce")
    reduce.dispatch_async(
        [partials, output, bias],
        rows * columns * batch,
        [rows * columns, columns, plan.splits, epilogue, batch]
    )
}

#: The plans a tuning pass times, likeliest winners first, so a pass cut short
#: by its budget has measured the ones that matter.
#:
#: The graph comes first, then the static plan, then the direct kernels. The
#: staged kernels are timed only in half precision, where they convert on the
#: way in, or when no direct kernel divides the shape: in float they ran at
#: 513-980 GFLOPS against 1.7-3.7 TFLOPS for direct and 5.8 for the graph,
#: and were half of every float pass.
pub fn candidates(
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false
) -> list[Plan] {
    var found: list[Plan] = []
    if mps_possible(rows, inner, columns, trans_a, trans_b) {
        found.push(MPS_PLAN)
        if gpu.mixed_precision() { found.push(MPS16_PLAN) }
    }
    let fallback = default_plan(rows, inner, columns, trans_a, trans_b)
    #: Where the static plan is the graph on a large output, only the graph's
    #: two precisions compete. Every tuned cache on record chose the graph for
    #: these, and the exceptions did not hold up: `direct32x32:1` cached for
    #: 1024x768x768 ran 0.29 ms against the graph's 0.21, and a direct kernel
    #: tuned for 4096x3072x1024 ran at two thirds of the graph's rate -- queued
    #: as one long command buffer, where the graph commits its own, it was also
    #: killed by the GPU watchdog.
    #:
    #: That holds when the graph reads the operands through padded copies
    #: too: the static plan takes the graph there only when the copies are
    #: small beside the product, and a staged kernel timed against a large
    #: ragged product runs a second or more per sample.
    #:
    #: On a small output the graph is only the static guess, and kernels are
    #: timed against it. Pinning it was a permanent loss on the shapes it
    #: guesses wrong: a 40000-deep gram at width 96 ran four times slower on
    #: the graph than on `direct32x32:16`, with no pass ever allowed to see it.
    if fallback.config < 0 {
        let graph_only = rows * inner * columns >= MPS_STATIC_WORK and
            not small_output(rows, columns)
        if graph_only { return found }
    } else {
        found.push(fallback)
    }
    var direct_legal = false
    for (index, config) in CONFIGS.enumerate() {
        if config.staged or not config_allowed(config, rows, inner, columns) { continue }
        direct_legal = true
        for splits in split_counts() {
            if not split_allowed(config, rows, inner, columns, splits) { continue }
            if index == fallback.config and splits == fallback.splits { continue }
            found.push(Plan(config.entry, index, splits))
        }
    }
    if gpu.mixed_precision() or not direct_legal {
        for (index, config) in CONFIGS.enumerate() {
            if not config.staged { continue }
            for splits in split_counts() {
                if not split_allowed(config, rows, inner, columns, splits) { continue }
                if index == fallback.config and splits == fallback.splits { continue }
                found.push(Plan(config.entry, index, splits))
            }
        }
    }
    return found
}

fn run_repeats(
    plan: Plan,
    a: Buffer,
    b: Buffer,
    output: Buffer,
    bias: Buffer,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool,
    trans_b: bool,
    repeats: int
) -> float {
    gpu.synchronize()
    let started = Instant.now()
    for _repeat in 0..repeats {
        encode(plan, a, b, output, bias, rows, inner, columns, trans_a, trans_b, 0)
    }
    gpu.synchronize()
    return started.elapsed().to_seconds()
}

#: Bring the clock up before anything is timed, and report what one call of
#: `plan` costs once it is up. Bounded both ways: enough dispatches for a small
#: shape, and a wall-clock cap so a large one does not spend longer warming
#: than measuring.
fn warm_clock(
    plan: Plan,
    a: Buffer,
    b: Buffer,
    output: Buffer,
    bias: Buffer,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool,
    trans_b: bool
) -> float {
    let opening = Instant.now()
    for round in 0..TUNE_WARMUP {
        encode(plan, a, b, output, bias, rows, inner, columns, trans_a, trans_b, 0)
        if round % TUNE_WARM_STRIDE == TUNE_WARM_STRIDE - 1 {
            gpu.synchronize()
            if opening.elapsed().to_seconds() >= TUNE_WARM_SECONDS { break }
        }
    }
    gpu.synchronize()
    let once = run_repeats(plan, a, b, output, bias, rows, inner, columns, trans_a, trans_b, 1)
    if once >= TUNE_MIN_SECONDS / 4.0 { return once }
    let eight = run_repeats(plan, a, b, output, bias, rows, inner, columns, trans_a, trans_b, 8)
    return eight / 8.0
}

var _last_candidates = 0
var _last_measured = 0

#: How many candidates the most recent tuning pass in this process had, and
#: how many of them it timed. For tests and profilers: a pass that timed fewer
#: than two of several chose nothing.
pub fn last_tune_counts() -> tuple[int, int] { return (_last_candidates, _last_measured) }

#: Time the candidates for one shape and return the fastest.
#:
#: Every candidate is timed in each of `TUNE_ROUNDS` rounds, the order turned
#: about each round, and keeps its fastest reading. One pass in a fixed order
#: hands the device's drift to whichever candidate went first: a plan is
#: measured once and believed until the epoch moves, and a run-off in a fixed
#: order picked the slower route for 4096x784x8192 even though the pass that
#: led to it had them the right way round. Each sample is a synchronised run of
#: the same repeat count for every candidate, sized from the static plan.
#:
#: Bounded by `TUNE_BUDGET_SECONDS` of wall clock, warm-up included: a pass
#: over every candidate cost 45.7 s for one 4096-cubed product. A candidate
#: more than `TUNE_ABANDON` times the best so far is dropped after its round,
#: and a kernel plan displaces the graph only by `TUNE_MPS_MARGIN`.
fn autotune(
    a: Buffer,
    b: Buffer,
    output: Buffer,
    bias: Buffer,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool,
    trans_b: bool
) -> Plan {
    let verbose = autotune_verbose()
    #: The caller's queue drains before the clock starts. A bucket can earn
    #: its pass in the middle of an asynchronous loop, and the pass's first
    #: wait is then for everything that loop queued: with more than the
    #: budget in flight, no candidate was ever timed, and the static plan was
    #: written down as the tuned one -- for every later process too.
    gpu.synchronize()
    let opening = Instant.now()
    let fallback = default_plan(rows, inner, columns, trans_a, trans_b)
    let plans = candidates(rows, inner, columns, trans_a, trans_b)
    _last_candidates = plans.len()
    _last_measured = 0
    if plans.len() <= 1 {
        if verbose {
            print(f"tune {rows}x{inner}x{columns} ta={trans_a} tb={trans_b} has one candidate")
        }
        return if plans.len() == 1 { plans[0] } else { fallback }
    }
    let each = warm_clock(fallback, a, b, output, bias, rows, inner, columns, trans_a, trans_b)
    var repeats = TUNE_MAX_REPEATS
    if each > 0.0 { repeats = max(1, min(TUNE_MAX_REPEATS, int(TUNE_MIN_SECONDS / each) + 1)) }
    var best_seen: list[float] = [-1.0 for _plan in plans]
    var alive: list[bool] = [true for _plan in plans]
    var out_of_time = false
    for round in 0..TUNE_MAX_ROUNDS {
        #: Rounds past the first `TUNE_ROUNDS` are taken only while a whole
        #: one still fits in the budget, so every survivor keeps the same
        #: number of readings.
        if round >= TUNE_ROUNDS {
            var round_cost = 0.0
            for index in 0..plans.len() {
                if alive[index] {
                    round_cost += best_seen[index] * float(repeats) + TUNE_SYNC_SECONDS
                }
            }
            if opening.elapsed().to_seconds() + round_cost > TUNE_BUDGET_SECONDS { break }
        }
        for step in 0..plans.len() {
            let index = if round % 2 == 0 { step } else { plans.len() - 1 - step }
            if not alive[index] { continue }
            if opening.elapsed().to_seconds() >= TUNE_BUDGET_SECONDS {
                out_of_time = true
                break
            }
            let plan = plans[index]
            var seconds = 0.0
            try {
                #: A first encode compiles; it is not what is being timed.
                if round == 0 {
                    encode(plan, a, b, output, bias, rows, inner, columns, trans_a, trans_b, 0)
                }
                seconds = run_repeats(plan, a, b, output, bias, rows, inner, columns, trans_a, trans_b, repeats) /
                    float(repeats)
            } catch _error: Error {
                alive[index] = false
                continue
            }
            if verbose {
                let gflops = float(2 * rows * inner * columns) / seconds / 1.0e9
                print(
                    f"tune {rows}x{inner}x{columns} ta={trans_a} tb={trans_b} r{round} {plan.to_str()} {gflops:.0f} GFLOPS"
                )
            }
            if best_seen[index] < 0.0 or seconds < best_seen[index] { best_seen[index] = seconds }
        }
        var fastest = -1.0
        for index in 0..plans.len() {
            if alive[index] and
                best_seen[index] >= 0.0 and
                (fastest < 0.0 or best_seen[index] < fastest) {
                fastest = best_seen[index]
            }
        }
        for index in 0..plans.len() {
            if best_seen[index] < 0.0 or best_seen[index] > fastest * TUNE_ABANDON {
                alive[index] = false
            }
        }
        if out_of_time { break }
    }
    var best = fallback
    var best_time = -1.0
    var graph = fallback
    var graph_time = -1.0
    for index in 0..plans.len() {
        let seconds = best_seen[index]
        if seconds < 0.0 { continue }
        _last_measured += 1
        if best_time < 0.0 or seconds < best_time {
            best = plans[index]
            best_time = seconds
        }
        if plans[index].config < 0 and (graph_time < 0.0 or seconds < graph_time) {
            graph = plans[index]
            graph_time = seconds
        }
    }
    if best.config >= 0 and graph_time > 0.0 and best_time * TUNE_MPS_MARGIN > graph_time {
        best = graph
    }
    if verbose {
        let spent = opening.elapsed().to_seconds() * 1000.0
        print(
            f"tune {rows}x{inner}x{columns} ta={trans_a} tb={trans_b} chose {best.to_str()} after {spent:.0f} ms, {repeats} repeats a sample"
        )
    }
    return best
}

#: The plan state a bucket of shapes shares.
class Bucket {
    #: Its row in the plan cache.
    pub let key: str
    #: The tuned plan, or null while the bucket runs on static plans.
    pub var plan: Plan?
    #: Tuned, read back from the cache, or tried and failed: never tune again.
    pub var settled: bool
    #: Passes that timed too few candidates to choose; see `tune_bucket`.
    pub var attempts: int
    #: Bumped whenever `plan` changes, so routes know to look again.
    pub var version: int

    pub fn init(self, key: str, plan: Plan?) {
        self.key = key
        self.plan = plan
        self.settled = plan is not null
        self.attempts = 0
        self.version = 0
    }
}

#: One exact shape: which bucket it belongs to and what it runs.
class Route {
    pub let rows: int
    pub let inner: int
    pub let columns: int
    pub let trans_a: bool
    pub let trans_b: bool
    pub let amp: bool
    pub let bucket: Bucket
    #: Never tuned: too small to be worth it, or tuning is off.
    pub let fixed: bool
    #: Estimated seconds one call costs; see `TUNE_PAYBACK_SECONDS`.
    pub let estimate: float
    #: Calls of this exact shape while its bucket had no plan.
    pub var requests: int
    #: Estimated seconds of work those calls did.
    pub var credit: float
    pub var plan: Plan
    pub var version: int

    pub fn init(
        self,
        rows: int,
        inner: int,
        columns: int,
        trans_a: bool,
        trans_b: bool,
        amp: bool,
        bucket: Bucket,
        fixed: bool,
        estimate: float,
        plan: Plan
    ) {
        self.rows = rows
        self.inner = inner
        self.columns = columns
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.amp = amp
        self.bucket = bucket
        self.fixed = fixed
        self.estimate = estimate
        self.requests = 0
        self.credit = 0.0
        self.plan = plan
        self.version = -1
    }

    pub fn matches(
        self,
        rows: int,
        inner: int,
        columns: int,
        trans_a: bool,
        trans_b: bool,
        amp: bool
    ) -> bool {
        return self.rows == rows and
            self.inner == inner and
            self.columns == columns and
            self.trans_a == trans_a and
            self.trans_b == trans_b and
            self.amp == amp
    }
}

var _routes: dict[int, Route] = {}
var _buckets: dict[str, Bucket] = {}

#: Forget every plan and request count this process holds, and read the plan
#: cache again -- from wherever `JAITENSOR_GEMM_CACHE` points now -- on the
#: next product. `JAITENSOR_GEMM_PLAN` is read again too.
#:
#: The cache is read once and every shape's route memoised, so pointing the
#: variable somewhere else mid-process changed nothing: the planner went on
#: believing the first file it read. A test of when tuning happens therefore
#: passed or failed on what the machine's own cache happened to hold. Call
#: this after changing the variable.
pub fn reload_plans() -> void {
    _forced_read = false
    _forced = null
    _last_candidates = 0
    _last_measured = 0
    _cache = {}
    _cache_loaded = false
    _cache_dirty = false
    _buckets = {}
    _routes = {}
    _memo_key = -1
    _memo = null
}

#: The plan this exact shape runs: the bucket's tuned plan where it is legal
#: for this member of the bucket, and the static plan otherwise.
fn resolve(route: Route) -> Plan {
    route.version = route.bucket.version
    let tuned = route.bucket.plan
    if tuned is not null and
        plan_legal(tuned, route.rows, route.inner, route.columns, route.trans_a, route.trans_b) {
        return tuned
    }
    return default_plan(route.rows, route.inner, route.columns, route.trans_a, route.trans_b)
}

#: Estimated seconds one call of this shape takes; see `TUNE_PAYBACK_SECONDS`.
fn estimated_seconds(rows: int, inner: int, columns: int, staged: bool) -> float {
    let flops = 2.0 * float(rows) * float(inner) * float(columns)
    let bytes = 4.0 *
        (float(rows) * float(inner) + float(inner) * float(columns) + float(rows) * float(columns))
    let rate = if staged { TUNE_ASSUMED_STAGED_FLOPS } else { TUNE_ASSUMED_FLOPS }
    return TUNE_ASSUMED_CALL_SECONDS + max(flops / rate, bytes / TUNE_ASSUMED_BYTES_PER_SECOND)
}

fn route_for(rows: int, inner: int, columns: int, trans_a: bool, trans_b: bool) -> Route {
    let amp = gpu.mixed_precision()
    let key_bits = memo_key(rows, inner, columns, trans_a, trans_b) * 2 + (if amp { 1 } else { 0 })
    let remembered = _memo
    if remembered is not null and
        _memo_key == key_bits and
        remembered.matches(rows, inner, columns, trans_a, trans_b, amp) {
        return remembered
    }
    let found = _routes.get(key_bits, null)
    if found is not null and found.matches(rows, inner, columns, trans_a, trans_b, amp) {
        _memo_key = key_bits
        _memo = found
        return found
    }
    load_cache()
    let key = shape_key(rows, inner, columns, trans_a, trans_b)
    var bucket = _buckets.get(key, null)
    if bucket is null {
        let made = Bucket(key, _cache.get(key, null))
        _buckets[key] = made
        bucket = made
    }
    let fixed = not autotune_enabled() or rows * inner * columns < TUNE_MIN_WORK
    let fallback = default_plan(rows, inner, columns, trans_a, trans_b)
    let staged = fallback.config >= 0 and CONFIGS[fallback.config].staged
    let route = Route(
        rows,
        inner,
        columns,
        trans_a,
        trans_b,
        amp,
        bucket,
        fixed,
        estimated_seconds(rows, inner, columns, staged),
        MPS_PLAN
    )
    route.plan = resolve(route)
    #: Routes are cheap to rebuild and a program can ask for any number of
    #: shapes; buckets, which carry the counts, are kept.
    if _routes.len() >= 4096 { _routes = {} }
    _routes[key_bits] = route
    _memo_key = key_bits
    _memo = route
    return route
}

#: Tune `bucket` on this shape, on numbers of this module's own making.
#:
#: The answer is written down against the bucket and believed from then on,
#: so it has to be a property of the shape and not of whatever the caller was
#: multiplying. It was not, once: a first matmul over a freshly allocated
#: buffer multiplies whatever was in that memory, and float32 bits taken at
#: random are full of denormals, infinities and NaNs. The half-precision route
#: casts all of that to zero and the float32 route does not, so on rubbish the
#: half route read faster than it can ever run, and the benchmark cached it
#: for two shapes where float32 is nearly twice as quick.
#:
#: `output` is left as the caller's: it is only written.
fn tune_bucket(
    bucket: Bucket,
    output: Buffer,
    bias: Buffer,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool,
    trans_b: bool
) -> void {
    bucket.settled = true
    let sample_a = Buffer(rows * inner)
    let sample_b = Buffer(inner * columns)
    sample_a.fill_uniform(-1.0, 1.0, 0x5EED)
    sample_b.fill_uniform(-1.0, 1.0, 0x5EED + 1)
    var plan = MPS_PLAN
    try {
        plan = autotune(sample_a, sample_b, output, bias, rows, inner, columns, trans_a, trans_b)
    } catch error: Error {
        sample_a.free()
        sample_b.free()
        throw error
    }
    sample_a.free()
    sample_b.free()
    #: A pass that timed fewer than two of several candidates chose nothing,
    #: and believing it would write the static plan down as measured, never
    #: to be looked at again. The bucket keeps its static plan and has to earn
    #: another pass; after `TUNE_MAX_ATTEMPTS` it stops trying for this
    #: process, still without writing anything.
    if _last_candidates > 1 and _last_measured < 2 {
        bucket.attempts += 1
        for route in _routes.values() {
            if route.bucket is bucket {
                route.requests = 0
                route.credit = 0.0
            }
        }
        bucket.settled = bucket.attempts >= TUNE_MAX_ATTEMPTS
        if autotune_verbose() {
            print(
                f"tune {rows}x{inner}x{columns} timed {_last_measured} of {_last_candidates}; not kept"
            )
        }
        return
    }
    bucket.plan = plan
    bucket.version += 1
    _cache[bucket.key] = plan
    _cache_dirty = true
    store_cache()
}

#: Route `rows x inner x columns`, tuning its bucket once it has earned it.
#:
#: `output` is used as the timing workload when a tuning pass runs, so it
#: holds undefined data when this returns; the caller runs the plan next.
pub fn plan_for(
    a: Buffer,
    b: Buffer,
    output: Buffer,
    bias: Buffer,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool,
    trans_b: bool
) -> Plan {
    let forced = forced_plan()
    if forced is not null {
        if forced.config < 0 and mps_possible(rows, inner, columns, trans_a, trans_b) {
            return forced
        }
        if forced.config >= 0 and config_allowed(CONFIGS[forced.config], rows, inner, columns) {
            return forced
        }
        #: Keyed on nothing: a forced run measures the plan it names, and a
        #: shape that cannot run it gets the static plan, never a tuned one.
        return default_plan(rows, inner, columns, trans_a, trans_b)
    }
    let route = route_for(rows, inner, columns, trans_a, trans_b)
    let bucket = route.bucket
    if not route.fixed and not bucket.settled {
        #: Earned by one exact shape, not by the bucket as a whole: a
        #: generation walks a bucket's row counts one after another, each for
        #: a few dozen calls, and pooled, sixteen of them earned passes whose
        #: plan -- often a graph, compiled again for every new row count --
        #: was never paid back: 6.1 s for a 768-wide, 12-layer generation of
        #: 128 tokens against 4.4 s on static plans. A shape that recurs earns
        #: it alone, and its plan is still borrowed by the rest of the bucket.
        route.requests += 1
        route.credit += route.estimate
        let earned = route.requests >= TUNE_MIN_REQUESTS and route.credit >= TUNE_PAYBACK_SECONDS
        if earned or not _LAZY_TUNE {
            tune_bucket(bucket, output, bias, rows, inner, columns, trans_a, trans_b)
        }
    }
    if route.version != bucket.version { route.plan = resolve(route) }
    return route.plan
}

#: The route a shape already has, without measuring anything.
#:
#: Reports the forced plan, else the bucket's tuned plan where it is legal for
#: the shape, else the static heuristic, so profilers can explain a dispatch
#: without changing it.
pub fn planned_backend(
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false
) -> str {
    let forced = forced_plan()
    if forced is not null {
        let runs = if forced.config < 0 {
            mps_possible(rows, inner, columns, trans_a, trans_b)
        } else {
            config_allowed(CONFIGS[forced.config], rows, inner, columns)
        }
        if runs { return forced.to_str() }
        return default_plan(rows, inner, columns, trans_a, trans_b).to_str()
    }
    load_cache()
    let cached = _cache.get(shape_key(rows, inner, columns, trans_a, trans_b), null)
    if cached is not null and plan_legal(cached, rows, inner, columns, trans_a, trans_b) {
        return cached.to_str()
    }
    return default_plan(rows, inner, columns, trans_a, trans_b).to_str()
}

#: Multiply `a` by `b` into `output`, fusing `bias` and an activation.
#:
#: `epilogue` is 0 for none, 1 for bias, 2 for bias and ReLU, 3 for bias and
#: sigmoid, 4 for bias and tanh.
pub fn dispatch(
    a: Buffer,
    b: Buffer,
    output: Buffer,
    bias: Buffer,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false,
    epilogue: int = 0
) -> void {
    if rows <= 0 or columns <= 0 { return }
    let plan = plan_for(a, b, output, bias, rows, inner, columns, trans_a, trans_b)
    encode(plan, a, b, output, bias, rows, inner, columns, trans_a, trans_b, epilogue)
}

#: Run one product on `plan`, whatever the planner would have chosen.
#:
#: For tests and probes that need one particular route -- the padded MPS
#: route on a ragged width, a deep split -- without going through the
#: process-wide `JAITENSOR_GEMM_PLAN`. Raises `ValueError` when the plan
#: cannot run the shape.
pub fn dispatch_plan(
    plan: Plan,
    a: Buffer,
    b: Buffer,
    output: Buffer,
    bias: Buffer,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false,
    epilogue: int = 0
) -> void {
    if rows <= 0 or columns <= 0 { return }
    if not plan_legal(plan, rows, inner, columns, trans_a, trans_b) {
        throw ValueError(f"plan {plan.to_str()} cannot run {rows}x{inner}x{columns}")
    }
    encode(plan, a, b, output, bias, rows, inner, columns, trans_a, trans_b, epilogue)
}

#: Split a batch no finer than the tuner split one matrix.
#:
#: Splits exist to fill an M2 Max when one matrix has too few tiles to do it,
#: and each one past the first buys that parallelism with a slice of a partial
#: buffer and a pass to sum it. A batch supplies the same parallelism already,
#: so carrying the tuned count into a batch multiplies the partials by the batch
#: for no more occupancy: attention at seq 1024 tuned to sixteen splits, and
#: twelve heads of that is a fifty-megabyte scratch buffer read twice.
#:
#: Dropping straight to one split is worse still -- 1.90 ms against 0.97 for the
#: same forward pass -- because the tuner picked sixteen having measured that
#: sixteen tiles do not fill the device. What the tuner established is a group
#: COUNT that works; the batch should reach the same count and stop. Sixteen
#: splits over twelve heads becomes two, for 384 groups and 6 MB of partials.
fn batched_splits(
    config: Config,
    rows: int,
    inner: int,
    columns: int,
    tuned: int,
    batch: int
) -> int {
    var chosen = 1
    while chosen < tuned and chosen * batch < tuned {
        if not split_allowed(config, rows, inner, columns, chosen * 2) { break }
        chosen = chosen * 2
    }
    return chosen
}

#: Multiply `batch` pairs of equally shaped matrices in one dispatch.
#:
#: Every matrix in the batch has the same shape and the same transposes, so the
#: tile a threadgroup owns and which matrix it belongs to come out of one
#: threadgroup index and the operands out of one buffer each, `stride_a`,
#: `stride_b` and `stride_out` elements apart. What this saves is the dispatch,
#: and at attention's shapes that is most of the cost: twelve heads times four
#: products times a split reduction was eighty-five dispatches for six
#: gigaflops of work, which the device finishes in under a millisecond.
#:
#: The output must be contiguous — `stride_out` is only honoured by the single
#: dispatch, while the split reduction assumes matrices `rows * columns` apart.
#: `epilogue` is not supported here: a bias would have to say whether it is
#: shared across the batch or one per matrix, and no caller needs either yet.
pub fn dispatch_batched(
    a: Buffer,
    b: Buffer,
    output: Buffer,
    bias: Buffer,
    rows: int,
    inner: int,
    columns: int,
    batch: int,
    stride_a: int,
    stride_b: int,
    trans_a: bool = false,
    trans_b: bool = false
) -> void {
    if rows <= 0 or columns <= 0 or batch <= 0 { return }
    if batch == 1 {
        dispatch(a, b, output, bias, rows, inner, columns, trans_a, trans_b)
        return
    }
    let tuned = plan_for(a, b, output, bias, rows, inner, columns, trans_a, trans_b)
    var plan = tuned
    if tuned.config >= 0 and tuned.splits > 1 {
        let splits = batched_splits(
            CONFIGS[tuned.config],
            rows,
            inner,
            columns,
            tuned.splits,
            batch
        )
        if splits != tuned.splits { plan = Plan(tuned.backend, tuned.config, splits) }
    }
    encode(
        plan,
        a,
        b,
        output,
        bias,
        rows,
        inner,
        columns,
        trans_a,
        trans_b,
        0,
        batch: batch,
        stride_a: stride_a,
        stride_b: stride_b,
        stride_out: rows * columns
    )
}

#: Compile every tile configuration so a first timed step pays no compiler cost.
pub fn warmup() -> void {
    let element = element_type()
    for config in CONFIGS { kernel_for(element, config.entry) }
    kernel_for(element, "split_reduce")
}
