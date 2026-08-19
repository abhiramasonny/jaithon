#: Matrix multiplication: one general simdgroup kernel family, measured routing.
#:
#: A single templated Metal kernel covers every shape, both transposes, a fused
#: bias/activation epilogue and split-K. It is instantiated once per tile
#: configuration; `plan_for` decides which instantiation (or MPSGraph) runs a
#: given shape by timing the candidates on the device the program is actually
#: using, and caches the answer on disk.
#:
#: Autotuning happens on the first call for a shape. `JAITENSOR_AUTOTUNE=0`
#: keeps the static heuristic instead, `JAITENSOR_AUTOTUNE_VERBOSE=1` prints
#: every candidate it timed, `JAITENSOR_GEMM_CACHE` overrides the cache path,
#: and `JAITENSOR_GEMM_PLAN` forces one plan for every shape.
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from std.io import Path
from std.os import env, home_dir
from std.time import Instant

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
    const uint slice = wid / tiles;
    const uint tile = wid % tiles;
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
                    if (epilogue != 0u) total += bias[c];
                    if (epilogue == 2u) total = max(total, 0.0f);
                    else if (epilogue == 3u) total = 1.0f / (1.0f + exp(-total));
                    else if (epilogue == 4u) total = tanh(total);
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
    const uint slice = wid / tiles;
    const uint tile = wid % tiles;
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
                if (epilogue != 0u) total += bias[c];
                if (epilogue == 2u) total = max(total, 0.0f);
                else if (epilogue == 3u) total = 1.0f / (1.0f + exp(-total));
                else if (epilogue == 4u) total = tanh(total);
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
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float total = 0.0f;
    for (uint s = 0; s < splits; ++s) total += partials[s * count + id];
    if (epilogue != 0u) total += bias[id % columns];
    if (epilogue == 2u) total = max(total, 0.0f);
    else if (epilogue == 3u) total = 1.0f / (1.0f + exp(-total));
    else if (epilogue == 4u) total = tanh(total);
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

#: Fewest output tiles a configuration may produce before a smaller tile wins.
let MIN_TILES = 16

#: Threadgroups a split is allowed to grow the dispatch to.
let TARGET_GROUPS = 256

#: Products smaller than this take the static plan; measuring costs more than
#: the whole multiplication and the timings would be noise anyway.
let TUNE_MIN_WORK = 4194304

#: A candidate is timed over enough repeats to run for at least this long, so
#: dispatch overhead cannot decide the winner for a short kernel.
let TUNE_MIN_SECONDS = 0.004

#: How many times each candidate is timed; the fastest reading is kept.
let TUNE_ROUNDS = 3

#: Ceiling on those repeats, so tuning a slow candidate stays bounded.
let TUNE_MAX_REPEATS = 64

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
    _forced_read = true
    let text = env("JAITENSOR_GEMM_PLAN")
    if text is not null { _forced = plan_from_text(text) }
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

#: The same graph asked to run in half precision.
#:
#: Whether the conversion pays depends on the shape rather than on any rule
#: worth writing down: it costs a pass over A, over B, and twice over the
#: result, and it saves on the multiply. Both are timed and the winner kept.
let MPS16_PLAN = Plan("mpsgraph16", -2, 1)

fn mps_possible(rows: int, inner: int, columns: int, trans_a: bool, trans_b: bool) -> bool {
    if not gpu.is_available() { return false }
    let a_columns = if trans_a { rows } else { inner }
    let b_columns = if trans_b { inner } else { columns }
    return a_columns % 4 == 0 and b_columns % 4 == 0 and columns % 4 == 0
}

fn tile_count(config: Config, rows: int, columns: int) -> int {
    let across = (columns + config.tile_columns - 1) // config.tile_columns
    let down = (rows + config.tile_rows - 1) // config.tile_rows
    return across * down
}

fn split_allowed(config: Config, rows: int, inner: int, columns: int, splits: int) -> bool {
    if splits == 1 { return true }
    if inner // splits < 64 { return false }
    return tile_count(config, rows, columns) * splits <= 8192
}

fn config_allowed(config: Config, rows: int, inner: int, columns: int) -> bool {
    if config.staged { return true }
    return rows % config.tile_rows == 0 and columns % config.tile_columns == 0 and inner % 8 == 0
}

#: The routing a shape gets without measuring anything.
#:
#: Takes the largest tile that still produces a few threadgroups, then splits K
#: until the device has enough of them to stay busy. Shrinking the tile instead
#: costs arithmetic intensity, so splitting is always tried first.
pub fn default_plan(rows: int, inner: int, columns: int) -> Plan {
    var chosen = 4
    for (index, config) in CONFIGS.enumerate() {
        if config.staged and tile_count(config, rows, columns) >= MIN_TILES {
            chosen = index
            break
        }
    }
    let config = CONFIGS[chosen]
    var splits = 1
    for candidate in SPLIT_CANDIDATES {
        if not split_allowed(config, rows, inner, columns, candidate) { continue }
        if tile_count(config, rows, columns) * candidate <= TARGET_GROUPS { splits = candidate }
    }
    return Plan(config.entry, chosen, splits)
}

var _cache: dict[str, Plan] = {}
var _cache_loaded = false
var _cache_dirty = false

fn cache_path() -> Path {
    let override = env("JAITENSOR_GEMM_CACHE")
    if override is not null { return Path(override) }
    return Path(home_dir()).join(".cache", "jaithon", "gemm-plans.tsv")
}

#: Bumped whenever the kernels or the tile configurations change.
#:
#: A cached plan is a measurement of a set of candidates that no longer exists
#: once that set changes, and nothing would ever re-measure it: the entry is
#: found, believed, and used forever. Making the epoch part of the key retires
#: every old entry the first time a changed build runs, which costs one tuning
#: pass and is the only thing that keeps a stale answer from outliving the
#: kernel it was about.
let TUNE_EPOCH = 2

fn shape_key(rows: int, inner: int, columns: int, trans_a: bool, trans_b: bool) -> str {
    let a_flag = if trans_a { 1 } else { 0 }
    let b_flag = if trans_b { 1 } else { 0 }
    let prefix = f"{gpu.device_name()}\tv{TUNE_EPOCH}\t{element_type()}"
    return f"{prefix}\t{rows}\t{inner}\t{columns}\t{a_flag}\t{b_flag}"
}

var _memo_key = 0
var _memo_amp = false
var _memo: Plan? = null

#: A training step asks for the same handful of shapes on every batch, so the
#: plan lookup is memoised on the shape itself: a full lookup builds a cache key
#: string, and doing that per dispatch cost more than encoding the kernel.
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
pub fn plan_from_text(text: str) -> Plan? {
    let parts = text.split(":")
    if parts.len() != 2 { return null }
    let splits = int(parts[1])
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
    epilogue: int
) -> void {
    if plan.config < 0 {
        gpu.matmul_async(
            a,
            b,
            output,
            rows,
            inner,
            columns,
            trans_a: trans_a,
            trans_b: trans_b,
            half: plan.config == -2
        )
        if epilogue != 0 {
            let kernel = kernel_for(element_type(), "split_reduce")
            kernel.dispatch_async(
                [output, output, bias],
                rows * columns,
                [rows * columns, columns, 1, epilogue]
            )
        }
        return
    }
    let config = CONFIGS[plan.config]
    let element = element_type()
    let kernel = kernel_for(element, config.entry)
    let a_flag = if trans_a { 1 } else { 0 }
    let b_flag = if trans_b { 1 } else { 0 }
    let width = config.group_size()
    let groups = tile_count(config, rows, columns) * plan.splits
    if plan.splits == 1 {
        kernel.dispatch_async(
            [a, b, output, bias],
            groups * width,
            [rows, inner, columns, a_flag, b_flag, epilogue, 1],
            group_size: width
        )
        return
    }
    let partials = scratch_for(rows * columns * plan.splits)
    kernel.dispatch_async(
        [a, b, partials, bias],
        groups * width,
        [rows, inner, columns, a_flag, b_flag, epilogue, plan.splits],
        group_size: width
    )
    let reduce = kernel_for(element, "split_reduce")
    reduce.dispatch_async(
        [partials, output, bias],
        rows * columns,
        [rows * columns, columns, plan.splits, epilogue]
    )
}

fn candidates(rows: int, inner: int, columns: int, trans_a: bool, trans_b: bool) -> list[Plan] {
    var found: list[Plan] = []
    if mps_possible(rows, inner, columns, trans_a, trans_b) {
        found.push(MPS_PLAN)
        if gpu.mixed_precision() { found.push(MPS16_PLAN) }
    }
    for (index, config) in CONFIGS.enumerate() {
        if not config_allowed(config, rows, inner, columns) { continue }
        for splits in SPLIT_CANDIDATES {
            if not split_allowed(config, rows, inner, columns, splits) { continue }
            found.push(Plan(config.entry, index, splits))
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

fn measure(
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
    encode(plan, a, b, output, bias, rows, inner, columns, trans_a, trans_b, 0)
    var repeats = 4
    var seconds = run_repeats(
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
        repeats
    )
    while seconds < TUNE_MIN_SECONDS and repeats < TUNE_MAX_REPEATS {
        repeats = repeats * 2
        seconds = run_repeats(
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
            repeats
        )
    }

    #: The fastest of several timings rather than one timing. Noise on a GPU
    #: only ever adds time — another program's work, a clock that has not
    #: settled, a queue that was not empty — so the smallest reading is the
    #: closest to what the kernel actually costs. One reading each let a
    #: candidate fifteen per cent slower win whichever run it happened to be
    #: measured on a quiet moment.
    for _again in 1..TUNE_ROUNDS {
        let again = run_repeats(
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
            repeats
        )
        if again < seconds { seconds = again }
    }
    return seconds / float(repeats)
}

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
    var best = default_plan(rows, inner, columns)
    var best_seconds = 0.0
    var measured = false
    for plan in candidates(rows, inner, columns, trans_a, trans_b) {
        var seconds = 0.0
        try {
            seconds = measure(plan, a, b, output, bias, rows, inner, columns, trans_a, trans_b)
        } catch _error: Error {
            continue
        }
        if verbose {
            let gflops = float(2 * rows * inner * columns) / seconds / 1.0e9
            print(
                f"tune {rows}x{inner}x{columns} ta={trans_a} tb={trans_b} {plan.to_str()} {gflops:.0f} GFLOPS"
            )
        }
        if not measured or seconds < best_seconds {
            best = plan
            best_seconds = seconds
            measured = true
        }
    }
    if verbose {
        print(f"tune {rows}x{inner}x{columns} ta={trans_a} tb={trans_b} chose {best.to_str()}")
    }
    return best
}

#: Route `rows x inner x columns`, measuring the candidates the first time.
#:
#: `a`, `b` and `output` are used as the timing workload, so `output` holds
#: undefined data when this returns; the caller runs the winning plan next.
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
    }
    let key_bits = memo_key(rows, inner, columns, trans_a, trans_b)
    let amp = gpu.mixed_precision()
    let remembered = _memo
    if remembered is not null and _memo_key == key_bits and _memo_amp == amp {
        return remembered
    }
    load_cache()
    let key = shape_key(rows, inner, columns, trans_a, trans_b)
    let cached = _cache.get(key, null)
    if cached is not null {
        _memo_key = key_bits
        _memo_amp = amp
        _memo = cached
        return cached
    }
    var plan = default_plan(rows, inner, columns)
    if autotune_enabled() and rows * inner * columns >= TUNE_MIN_WORK {
        plan = autotune(a, b, output, bias, rows, inner, columns, trans_a, trans_b)
        _cache_dirty = true
    }
    _cache[key] = plan
    _memo_key = key_bits
    _memo_amp = amp
    _memo = plan
    store_cache()
    return plan
}

#: The route a shape already has, without measuring anything.
#:
#: Reports the cached plan when the shape has been tuned, and the static
#: heuristic otherwise, so profilers can explain a dispatch without changing it.
pub fn planned_backend(
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false
) -> str {
    load_cache()
    let cached = _cache.get(shape_key(rows, inner, columns, trans_a, trans_b), null)
    if cached is not null { return cached.to_str() }
    return default_plan(rows, inner, columns).to_str()
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

#: Compile every tile configuration so a first timed step pays no compiler cost.
pub fn warmup() -> void {
    let element = element_type()
    for config in CONFIGS { kernel_for(element, config.entry) }
    kernel_for(element, "split_reduce")
}
