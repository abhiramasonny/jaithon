#: Long/wide restructuring: pivot, melt, stack, crosstab and the rest.
#:
#: Every one of these is the same shape underneath. A reshape moves a value
#: from the row it is on to the cell a pair of keys names, so it is a
#: factorization of the keys followed by a scatter, and the interesting part is
#: that neither step ever looks at a cell that has no value in it. Writing
#: `pivot` as a loop over `rows * columns` cells asking "which input row lands
#: here" is the obvious implementation and it is quadratic; writing it as a loop
#: over input rows asking "which cell do I land in" is one dispatch over the
#: data.
#:
#: So the vocabulary here is small:
#:
#: - factorize a key into dense codes `0 .. distinct - 1`, on the device, in
#:   first-appearance order — the order `Column.from_strs` already numbers its
#:   levels in. A caller that wants the labels sorted sorts them; sorting is
#:   not this module's business.
#: - turn a `(row code, column code)` pair into one cell index, and scatter.
#: - slice the flat `rows * columns` result into one `Column` per output column.
#:
#: `pivot` refuses a duplicate `(row, column)` pair rather than picking a
#: winner, because a silent pick is a wrong answer that looks like a right one.
#: `pivot_table` is the same scatter with an aggregation in front of it, which
#: is what makes duplicates meaningful.
#:
#: ```
#: let long = pd.DataFrame(
#:     ["city", "year", "high"],
#:     [
#:         pd.Column.from_strs(["austin", "austin", "dallas"]),
#:         pd.Column.from_ints([2024, 2025, 2024]),
#:         pd.Column.from_floats([38.0, 39.5, 41.0]),
#:     ]
#: )
#: let wide = reshape.pivot(long, ["city"], ["year"], values: ["high"])
#: wide.names                    # ["2024", "2025"]
#: ```
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from .column import Agg, Column, DataFrame, Dtype, Index, Series, Value
from .column import MAX_ROWS, MISSING_POSITION, WIDE_LIMIT
from .column import cached_kernel, concat_columns, f32_bits, prefix_sum, wide_pack

#: How many rows `transpose` will turn into columns.
#:
#: A transposed frame has one column per input row, and a frame's columns are a
#: host list with a host name apiece. Past a few thousand that list costs more
#: than the transpose it describes, so the refusal is the honest answer rather
#: than a call that takes a minute.
pub const MAX_TRANSPOSE_ROWS = 4096

const AGG_SUM = 0
const AGG_MEAN = 1
const AGG_COUNT = 2
const AGG_SIZE = 3
const AGG_MIN = 4
const AGG_MAX = 5
const AGG_FIRST = 6
const AGG_LAST = 7
const AGG_ANY = 8
const AGG_ALL = 9

const EMPTY_SLOT = 4294967295

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant long WIDE_SCALE = 16777216L;

constexpr constant uint AGG_SUM = 0u;
constexpr constant uint AGG_MEAN = 1u;
constexpr constant uint AGG_COUNT = 2u;
constexpr constant uint AGG_SIZE = 3u;
constexpr constant uint AGG_MIN = 4u;
constexpr constant uint AGG_MAX = 5u;
constexpr constant uint AGG_FIRST = 6u;
constexpr constant uint AGG_LAST = 7u;
constexpr constant uint AGG_ANY = 8u;

inline void wide_store(device float* low, device float* high, uint at, long value) {
    long hi = value >> 24;
    low[at] = (float)(value - (hi << 24));
    high[at] = (float)hi;
}

/* float32 reordered into a uint that compares the same way, so that a minimum
 * and a maximum can run on `atomic_fetch_min`/`max`. Metal has no float
 * atomic minimum, and a compare-and-swap loop over a whole column is slower
 * than this by the width of the contention on a popular cell. */
inline uint order_bits(float value) {
    uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

/* Two rows carry the same key when both are missing, or when both lanes agree.
 *
 * NaN is compared equal to NaN on purpose: `col_hash` already folds every NaN
 * pattern to one hash, so leaving them unequal here would send them round the
 * probe forever looking for a slot that can never match. */
inline bool key_equal(device const float* values,
                      device const float* wide,
                      device const float* valid,
                      uint has_wide,
                      uint has_valid,
                      uint a,
                      uint b) {
    bool a_null = has_valid != 0u && valid[a] == 0.0f;
    bool b_null = has_valid != 0u && valid[b] == 0.0f;
    if (a_null || b_null) return a_null && b_null;
    float va = values[a];
    float vb = values[b];
    bool same = va == vb || (isnan(va) && isnan(vb));
    if (same && has_wide != 0u) same = wide[a] == wide[b];
    return same;
}

/* Linear probing with one compare-and-swap per insertion, over a table of
 * `mask + 1` slots holding `row + 1` so that a zeroed allocation reads as
 * empty and no fill kernel is needed.
 *
 * The slot ends up holding the *lowest* row carrying its key, which is what
 * makes the codes come out in first-appearance order two dispatches later.
 * Lowering it while other threads are still probing is safe because every row
 * in a slot carries the same key, so a comparison against whichever one is
 * there right now gives the same answer. */
kernel void frame_factorize_probe(device const uint* hashes [[buffer(0)]],
                                  device const float* values [[buffer(1)]],
                                  device const float* wide [[buffer(2)]],
                                  device const float* valid [[buffer(3)]],
                                  device atomic_uint* table [[buffer(4)]],
                                  device uint* slot_of_row [[buffer(5)]],
                                  constant uint& count [[buffer(6)]],
                                  constant uint& mask [[buffer(7)]],
                                  constant uint& has_wide [[buffer(8)]],
                                  constant uint& has_valid [[buffer(9)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint slot = hashes[gid] & mask;
    uint steps = 0u;
    while (steps <= mask) {
        uint seen = atomic_load_explicit(&table[slot], memory_order_relaxed);
        if (seen == 0u) {
            uint expected = 0u;
            bool won = atomic_compare_exchange_weak_explicit(&table[slot],
                                                             &expected,
                                                             gid + 1u,
                                                             memory_order_relaxed,
                                                             memory_order_relaxed);
            if (won) break;
            /* Re-read the same slot rather than moving on. A weak exchange may
             * fail spuriously with the slot still empty, and advancing on that
             * would put one key in two slots. */
            continue;
        }
        if (key_equal(values, wide, valid, has_wide, has_valid, seen - 1u, gid)) break;
        slot = (slot + 1u) & mask;
        steps++;
    }
    slot_of_row[gid] = slot;
    atomic_fetch_min_explicit(&table[slot], gid + 1u, memory_order_relaxed);
}

kernel void frame_factorize_mark(device const uint* table [[buffer(0)]],
                                 device const uint* slot_of_row [[buffer(1)]],
                                 device float* owner [[buffer(2)]],
                                 device float* first [[buffer(3)]],
                                 constant uint& count [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint holder = table[slot_of_row[gid]] - 1u;
    owner[gid] = (float)holder;
    first[gid] = holder == gid ? 1.0f : 0.0f;
}

/* The scan of the first-occurrence flags is already the code: the number of
 * distinct keys that first appeared before a row is that row's key's rank. */
kernel void frame_factorize_codes(device const float* owner [[buffer(0)]],
                                  device const float* first [[buffer(1)]],
                                  device const float* offsets [[buffer(2)]],
                                  device float* codes [[buffer(3)]],
                                  device float* first_rows [[buffer(4)]],
                                  constant uint& count [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float code = offsets[(uint)owner[gid]];
    codes[gid] = code;
    if (first[gid] != 0.0f) first_rows[(uint)code] = (float)gid;
}

/* Mixed-radix accumulation of one more key's codes, in the two-lane integer
 * encoding so that a product of cardinalities up to 2^48 stays exact. */
kernel void frame_combine_codes(device float* low [[buffer(0)]],
                                device float* high [[buffer(1)]],
                                device const float* codes [[buffer(2)]],
                                constant uint& count [[buffer(3)]],
                                constant uint& cardinality [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    long carried = (long)high[gid] * WIDE_SCALE + (long)low[gid];
    wide_store(low, high, gid, carried * (long)cardinality + (long)codes[gid]);
}

/* The same accumulation in one float lane, for the unobserved-category path
 * where the product of the level counts is bounded by MAX_ROWS and so is
 * exact in float32 without the second lane. */
kernel void frame_combine_dense(device float* codes [[buffer(0)]],
                                device const float* add [[buffer(1)]],
                                constant uint& count [[buffer(2)]],
                                constant uint& cardinality [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    codes[gid] = codes[gid] * (float)cardinality + add[gid];
}

/* One digit of a mixed-radix cell number: one division to drop the levels
 * below it and one more for the digit itself, with the remainder taken by a
 * subtraction rather than a second modulo. */
kernel void frame_unstride(device float* positions [[buffer(0)]],
                           constant uint& count [[buffer(1)]],
                           constant uint& inner [[buffer(2)]],
                           constant uint& cardinality [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint above = gid / inner;
    positions[gid] = (float)(above - (above / cardinality) * cardinality);
}

kernel void frame_fill_u32(device uint* out [[buffer(0)]],
                           constant uint& count [[buffer(1)]],
                           constant uint& value [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = value;
}

kernel void frame_and_valid(device const float* valid [[buffer(0)]],
                            device float* mask [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (valid[gid] == 0.0f) mask[gid] = 0.0f;
}

/* The whole of `pivot`: every input row claims its cell and leaves its row
 * number there, and the cells nobody claimed keep MISSING_POSITION, which
 * `take(allow_missing: true)` turns into nulls.
 *
 * The claim counter is what makes a duplicate pair an error rather than a
 * coin flip. Every loser writes the same 1.0 into `conflict`, so that race is
 * benign and the host reads one float instead of scanning a lane. */
kernel void frame_pivot_scatter(device const float* row_codes [[buffer(0)]],
                                device const float* col_codes [[buffer(1)]],
                                device atomic_uint* claim [[buffer(2)]],
                                device float* positions [[buffer(3)]],
                                device float* conflict [[buffer(4)]],
                                constant uint& count [[buffer(5)]],
                                constant uint& rows [[buffer(6)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint cell = (uint)col_codes[gid] * rows + (uint)row_codes[gid];
    uint taken = atomic_fetch_add_explicit(&claim[cell], 1u, memory_order_relaxed);
    if (taken != 0u) {
        conflict[0] = 1.0f;
        return;
    }
    positions[cell] = (float)gid;
}

inline void agg_update(device atomic_float* acc,
                       device atomic_uint* aux,
                       uint cell,
                       uint op,
                       float value,
                       uint row) {
    if (op == AGG_SUM || op == AGG_MEAN) {
        atomic_fetch_add_explicit(&acc[cell], value, memory_order_relaxed);
        atomic_fetch_add_explicit(&aux[cell], 1u, memory_order_relaxed);
        return;
    }
    if (op == AGG_COUNT || op == AGG_SIZE) {
        atomic_fetch_add_explicit(&aux[cell], 1u, memory_order_relaxed);
        return;
    }
    /* Everything below reduces into `aux` in a way that cannot distinguish an
     * empty cell from an occupied one, so `acc` carries the occupancy. */
    atomic_fetch_add_explicit(&acc[cell], 1.0f, memory_order_relaxed);
    if (op == AGG_MIN) {
        atomic_fetch_min_explicit(&aux[cell], order_bits(value), memory_order_relaxed);
    } else if (op == AGG_MAX) {
        atomic_fetch_max_explicit(&aux[cell], order_bits(value), memory_order_relaxed);
    } else if (op == AGG_FIRST) {
        atomic_fetch_min_explicit(&aux[cell], row + 1u, memory_order_relaxed);
    } else if (op == AGG_LAST) {
        atomic_fetch_max_explicit(&aux[cell], row + 1u, memory_order_relaxed);
    } else if (op == AGG_ANY) {
        atomic_fetch_max_explicit(&aux[cell], value != 0.0f ? 1u : 0u, memory_order_relaxed);
    } else {
        atomic_fetch_min_explicit(&aux[cell], value != 0.0f ? 1u : 0u, memory_order_relaxed);
    }
}

/* Margins cost three more atomic updates per row rather than a second pass
 * over the data, and they are correct for every reduction here because each
 * one is associative and the margin cell simply sees more of the rows. */
kernel void frame_agg_cells(device const float* lane [[buffer(0)]],
                            device const float* valid [[buffer(1)]],
                            device const float* row_codes [[buffer(2)]],
                            device const float* col_codes [[buffer(3)]],
                            device atomic_float* acc [[buffer(4)]],
                            device atomic_uint* aux [[buffer(5)]],
                            constant uint& count [[buffer(6)]],
                            constant uint& rows [[buffer(7)]],
                            constant uint& cols [[buffer(8)]],
                            constant uint& op [[buffer(9)]],
                            constant uint& has_valid [[buffer(10)]],
                            constant uint& margins [[buffer(11)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool present = has_valid == 0u || valid[gid] != 0.0f;
    if (!present && op != AGG_SIZE) return;
    float value = lane[gid];
    uint height = rows + (margins != 0u ? 1u : 0u);
    uint r = (uint)row_codes[gid];
    uint c = (uint)col_codes[gid];
    agg_update(acc, aux, c * height + r, op, value, gid);
    if (margins == 0u) return;
    agg_update(acc, aux, c * height + rows, op, value, gid);
    agg_update(acc, aux, cols * height + r, op, value, gid);
    agg_update(acc, aux, cols * height + rows, op, value, gid);
}

/* The row that carried a cell's extremum, lowest wins a tie.
 *
 * A second pass rather than a wider accumulator: there is no 64-bit atomic to
 * hold the value and the row together, and a compare-and-swap loop that keeps
 * the pair consistent serialises on exactly the cells that have the most rows
 * in them. */
kernel void frame_agg_pick_row(device const float* lane [[buffer(0)]],
                               device const float* valid [[buffer(1)]],
                               device const float* row_codes [[buffer(2)]],
                               device const float* col_codes [[buffer(3)]],
                               device const uint* aux [[buffer(4)]],
                               device atomic_uint* chosen [[buffer(5)]],
                               constant uint& count [[buffer(6)]],
                               constant uint& rows [[buffer(7)]],
                               constant uint& cols [[buffer(8)]],
                               constant uint& has_valid [[buffer(9)]],
                               constant uint& margins [[buffer(10)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (has_valid != 0u && valid[gid] == 0.0f) return;
    uint bits = order_bits(lane[gid]);
    uint height = rows + (margins != 0u ? 1u : 0u);
    uint r = (uint)row_codes[gid];
    uint c = (uint)col_codes[gid];
    uint cell = c * height + r;
    if (aux[cell] == bits) atomic_fetch_min_explicit(&chosen[cell], gid + 1u, memory_order_relaxed);
    if (margins == 0u) return;
    cell = c * height + rows;
    if (aux[cell] == bits) atomic_fetch_min_explicit(&chosen[cell], gid + 1u, memory_order_relaxed);
    cell = cols * height + r;
    if (aux[cell] == bits) atomic_fetch_min_explicit(&chosen[cell], gid + 1u, memory_order_relaxed);
    cell = cols * height + rows;
    if (aux[cell] == bits) atomic_fetch_min_explicit(&chosen[cell], gid + 1u, memory_order_relaxed);
}

kernel void frame_agg_positions(device const float* acc [[buffer(0)]],
                                device const uint* chosen [[buffer(1)]],
                                device float* positions [[buffer(2)]],
                                constant uint& cells [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= cells) return;
    uint picked = chosen[gid];
    bool occupied = acc[gid] > 0.0f && picked != 0u && picked != 0xFFFFFFFFu;
    positions[gid] = occupied ? (float)(picked - 1u) : -1.0f;
}

kernel void frame_agg_finalize(device const float* acc [[buffer(0)]],
                               device const uint* aux [[buffer(1)]],
                               device float* out_values [[buffer(2)]],
                               device float* out_wide [[buffer(3)]],
                               device float* out_valid [[buffer(4)]],
                               constant uint& cells [[buffer(5)]],
                               constant uint& op [[buffer(6)]],
                               constant uint& to_wide [[buffer(7)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= cells) return;
    uint n = aux[gid];
    float total = acc[gid];
    bool present;
    float result;
    if (op == AGG_SUM) {
        present = n > 0u;
        result = total;
    } else if (op == AGG_MEAN) {
        present = n > 0u;
        result = present ? total / (float)n : 0.0f;
    } else if (op == AGG_COUNT || op == AGG_SIZE) {
        present = n > 0u;
        result = (float)n;
    } else {
        present = total > 0.0f;
        result = present ? (float)n : 0.0f;
    }
    out_valid[gid] = present ? 1.0f : 0.0f;
    if (to_wide != 0u) {
        wide_store(out_values, out_wide, gid, present ? (long)result : 0L);
        return;
    }
    out_values[gid] = present ? result : 0.0f;
}

/* Written into a second lane, not in place: the divisors are cells of the same
 * grid, and a thread that normalised its own margin before its neighbour read
 * it would hand that neighbour a ratio of ratios. */
kernel void frame_normalize_cells(device const float* src [[buffer(0)]],
                                  device float* dst [[buffer(1)]],
                                  constant uint& cells [[buffer(2)]],
                                  constant uint& height [[buffer(3)]],
                                  constant uint& width [[buffer(4)]],
                                  constant uint& mode [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= cells) return;
    uint c = gid / height;
    uint r = gid - c * height;
    float divisor;
    if (mode == 0u) divisor = src[(width - 1u) * height + (height - 1u)];
    else if (mode == 1u) divisor = src[(width - 1u) * height + r];
    else divisor = src[c * height + (height - 1u)];
    dst[gid] = divisor != 0.0f ? src[gid] / divisor : 0.0f;
}

kernel void frame_fill_nulls(device float* values [[buffer(0)]],
                             device float* wide [[buffer(1)]],
                             device float* valid [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             constant uint& has_wide [[buffer(4)]],
                             constant uint& fill_low [[buffer(5)]],
                             constant int& fill_high [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (valid[gid] != 0.0f) return;
    valid[gid] = 1.0f;
    if (has_wide != 0u) {
        values[gid] = (float)fill_low;
        wide[gid] = (float)fill_high;
        return;
    }
    values[gid] = as_type<float>(fill_low);
}

/* melt and wide_to_long: block `b` of the output repeats every input row, so
 * the source row is the offset within the block and the block number is the
 * variable's code. One division, remainder by subtraction. */
kernel void frame_tile_positions(device float* positions [[buffer(0)]],
                                 device float* blocks [[buffer(1)]],
                                 constant uint& count [[buffer(2)]],
                                 constant uint& rows [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint block = gid / rows;
    positions[gid] = (float)(gid - block * rows);
    blocks[gid] = (float)block;
}

/* stack: the output runs row-major over a column-major staging column, so the
 * gather stride is what turns one layout into the other. */
kernel void frame_stack_positions(device float* value_positions [[buffer(0)]],
                                  device float* row_positions [[buffer(1)]],
                                  device float* column_codes [[buffer(2)]],
                                  constant uint& count [[buffer(3)]],
                                  constant uint& cols [[buffer(4)]],
                                  constant uint& rows [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint row = gid / cols;
    uint col = gid - row * cols;
    value_positions[gid] = (float)(col * rows + row);
    row_positions[gid] = (float)row;
    column_codes[gid] = (float)col;
}

kernel void frame_transpose_positions(device float* positions [[buffer(0)]],
                                      constant uint& count [[buffer(1)]],
                                      constant uint& cols [[buffer(2)]],
                                      constant uint& rows [[buffer(3)]],
                                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint out_col = gid / cols;
    uint out_row = gid - out_col * cols;
    positions[gid] = (float)(out_row * rows + out_col);
}

/* get_dummies: one thread per output slot across every indicator column at
 * once, so the whole expansion is a single dispatch over `levels * rows`. */
kernel void frame_dummy_indicators(device const float* codes [[buffer(0)]],
                                   device const float* valid [[buffer(1)]],
                                   device float* out [[buffer(2)]],
                                   device float* out_valid [[buffer(3)]],
                                   constant uint& count [[buffer(4)]],
                                   constant uint& rows [[buffer(5)]],
                                   constant uint& base [[buffer(6)]],
                                   constant uint& has_valid [[buffer(7)]],
                                   constant uint& dst_has_valid [[buffer(8)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint level = gid / rows;
    uint row = gid - level * rows;
    bool present = has_valid == 0u || valid[row] != 0.0f;
    out[gid] = present && (uint)codes[row] == level + base ? 1.0f : 0.0f;
    if (dst_has_valid != 0u) out_valid[gid] = present ? 1.0f : 0.0f;
}

/* The same expansion four rows at a time. The innermost run of the output is
 * the rows within one indicator column, so a row count divisible by four makes
 * both the code read and the indicator write a float4. */
kernel void frame_dummy_indicators4(device const float4* codes [[buffer(0)]],
                                    device const float4* valid [[buffer(1)]],
                                    device float4* out [[buffer(2)]],
                                    device float4* out_valid [[buffer(3)]],
                                    constant uint& quads [[buffer(4)]],
                                    constant uint& row_quads [[buffer(5)]],
                                    constant uint& base [[buffer(6)]],
                                    constant uint& has_valid [[buffer(7)]],
                                    constant uint& dst_has_valid [[buffer(8)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= quads) return;
    uint level = gid / row_quads;
    uint quad = gid - level * row_quads;
    float4 present = has_valid == 0u
        ? float4(1.0f)
        : select(float4(0.0f), float4(1.0f), valid[quad] != 0.0f);
    float4 hit = select(float4(0.0f), float4(1.0f), codes[quad] == (float)(level + base));
    out[gid] = hit * present;
    if (dst_has_valid != 0u) out_valid[gid] = present;
}

kernel void frame_explode_counts(device const float* codes [[buffer(0)]],
                                 device const float* valid [[buffer(1)]],
                                 device const float* per_level [[buffer(2)]],
                                 device float* counts [[buffer(3)]],
                                 constant uint& count [[buffer(4)]],
                                 constant uint& levels [[buffer(5)]],
                                 constant uint& has_valid [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (has_valid != 0u && valid[gid] == 0.0f) {
        counts[gid] = 1.0f;
        return;
    }
    int code = (int)codes[gid];
    counts[gid] = code < 0 || code >= (int)levels ? 1.0f : per_level[code];
}

kernel void frame_scatter_starts(device const float* starts [[buffer(0)]],
                                 device float* flags [[buffer(1)]],
                                 constant uint& count [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    flags[(uint)starts[gid]] = 1.0f;
}

/* An inclusive scan of the run-start flags is the source row of every output
 * slot, which is how a variable-length expansion is inverted without a search
 * per output row. */
kernel void frame_expand_rows(device const float* scanned [[buffer(0)]],
                              device const float* flags [[buffer(1)]],
                              device float* source_rows [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    source_rows[gid] = scanned[gid] + flags[gid] - 1.0f;
}

kernel void frame_explode_codes(device const float* source_rows [[buffer(0)]],
                                device const float* starts [[buffer(1)]],
                                device const float* codes [[buffer(2)]],
                                device const float* level_start [[buffer(3)]],
                                device const float* parts [[buffer(4)]],
                                device const float* valid [[buffer(5)]],
                                device float* out_codes [[buffer(6)]],
                                device float* out_valid [[buffer(7)]],
                                constant uint& count [[buffer(8)]],
                                constant uint& has_valid [[buffer(9)]],
                                constant uint& dst_has_valid [[buffer(10)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint row = (uint)source_rows[gid];
    bool present = has_valid == 0u || valid[row] != 0.0f;
    if (dst_has_valid != 0u) out_valid[gid] = present ? 1.0f : 0.0f;
    if (!present) {
        out_codes[gid] = 0.0f;
        return;
    }
    uint at = (uint)level_start[(uint)codes[row]] + (uint)((float)gid - starts[row]);
    out_codes[gid] = parts[at];
}
"""

var _kernels: dict[str, Kernel] = {}

#: The compiled pipeline for one entry point in the reshaping shader library.
fn reshape_kernel(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, entry)
    _kernels[entry] = kernel
    return kernel
}

fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

fn u32(value: int) -> int { return value & 0xFFFFFFFF }

fn require_device(what: str) -> void {
    if not gpu.is_available() {
        throw RuntimeError(f"jaiframe needs a GPU to {what}")
    }
}

fn lane_copy(
    src: Buffer,
    dst: Buffer,
    count: int,
    src_offset: int = 0,
    dst_offset: int = 0
) -> void {
    if count == 0 { return }
    cached_kernel("col_copy_lane").dispatch_async(
        [src, dst],
        count,
        [count, src_offset, dst_offset]
    )
}

fn filled_lane(count: int, value: float) -> Buffer {
    let lane = Buffer(count)
    if count > 0 { lane.fill(value) }
    return lane
}

#: A lane of `count` uints all holding `value`.
#:
#: `Buffer.fill` writes floats, and the sentinel a minimum starts from is
#: 0xFFFFFFFF, which is a NaN bit pattern rather than a number.
fn filled_u32_lane(count: int, value: int) -> Buffer {
    let lane = Buffer(count)
    if count == 0 { return lane }
    if value == 0 {
        lane.fill_zero()
        return lane
    }
    reshape_kernel("frame_fill_u32").dispatch_async([lane], count, [count, u32(value)])
    return lane
}

#: The smallest power of two at least `value`, floored at 16.
#:
#: The hash table is sized from this and probed with a mask rather than a
#: modulo, because the device has no integer divide and a factorization does
#: one probe step per collision.
fn next_power_of_two(value: int) -> int {
    var size = 16
    for _step in 0..32 {
        if size >= value { return size }
        size = size * 2
    }
    throw ValueError(f"{value} keys is more than a hash table can be sized for")
}

#: One host string per row, whatever the column holds.
#:
#: Output column names and margin labels are host strings, so a reshape has to
#: cross back at some point. It crosses here, once per label column, over the
#: distinct labels rather than over the rows — a pivot with a million rows and
#: twelve columns formats twelve strings.
fn label_strings(column: Column) -> list[str] {
    if column.dtype.is_dictionary() { return column.to_strs().map(|text| text ?? "NA") }
    if column.dtype.is_wide() { return column.to_ints().map(|value| f"{value}") }
    if column.dtype == Dtype.Bool {
        return column.to_bools().map(|value| if value { "true" } else { "false" })
    }
    return column.to_floats().map(|value| f"{value}")
}

#: The label of each row across several level columns, joined with `_`.
#:
#: A frame's column names are flat strings, so a pivot on two key columns has
#: to spell the pair as one name. pandas would build a MultiIndex over the
#: column axis; that is a second index class to carry through every module for
#: the sake of a name, and a joined name reads the same in a table.
fn joined_labels(levels: list[Column], count: int) -> list[str] {
    if levels.len() == 0 { return [f"{at}" for at in 0..count] }
    let parts = levels.map(|level| label_strings(level))
    var out: list[str] = []
    for at in 0..count {
        var pieces: list[str] = []
        for part in parts { pieces.push(part[at]) }
        out.push("_".join(pieces))
    }
    return out
}

fn optional_names(names: list[str]) -> list[str?] {
    var out: list[str?] = []
    for name in names { out.push(name) }
    return out
}

#: One column per contiguous run of `rows` slots in a flat grid.
#:
#: The scatter writes the whole `rows * columns` result into one allocation,
#: because a scatter that has to pick which of several buffers to write into is
#: a scatter with a branch per row. Splitting it afterwards costs one copy of
#: the output and a dispatch per output column, not per row.
fn slice_grid(flat: Column, rows: int, count: int, first: int = 0) -> list[Column] {
    var out: list[Column] = []
    for at in 0..count {
        let part = Column.allocate(flat.dtype, rows, with_validity: flat.has_validity())
        if flat.dtype.is_dictionary() { part.set_levels([name for name in flat.levels ?? []]) }
        let start = (first + at) * rows
        lane_copy(flat.values, part.values, rows, start, 0)
        let high = flat.wide
        let part_high = part.wide
        if high is not null and part_high is not null { lane_copy(high, part_high, rows, start, 0) }
        let mask = flat.valid
        let part_mask = part.valid
        if mask is not null and part_mask is not null { lane_copy(mask, part_mask, rows, start, 0) }
        out.push(part)
    }
    return out
}

#: The two scalars `frame_fill_nulls` needs to write one cell of any dtype.
fn fill_scalars(column: Column, value: Value) -> (int, int) {
    if column.dtype.is_wide() {
        let split = wide_pack(value.to_int())
        return (int(split[0]), u32(int(split[1])))
    }
    if column.dtype.is_dictionary() {
        return (f32_bits(float(column.intern_level(value.to_str()))), 0)
    }
    if column.dtype == Dtype.Bool {
        return (f32_bits(if value.to_bool() { 1.0 } else { 0.0 }), 0)
    }
    return (f32_bits(value.to_float()), 0)
}

#: Replace every missing cell with `value`, in place.
#:
#: pandas' `fill_value` applies after the reshape, and so does this: a cell is
#: missing because no input row landed in it, which is a fact about the data
#: that the aggregation should see before the filler does.
fn apply_fill(column: Column, value: Value) -> void {
    if column.length == 0 or value.is_na() { return }
    let mask = column.valid
    if mask is null { return }
    let scalars = fill_scalars(column, value)
    reshape_kernel("frame_fill_nulls").dispatch_async(
        [column.values, column.wide ?? column.values, mask],
        column.length,
        [column.length, flag(column.dtype.is_wide()), scalars[0], scalars[1]]
    )
}

#: True where every key column holds a value, or `null` when none can be null.
fn present_mask(keys: list[Column], length: int) -> Column? {
    var any = false
    for key in keys {
        if key.has_validity() { any = true }
    }
    if not any { return null }
    let mask = Column.allocate(Dtype.Bool, length)
    if length > 0 { mask.values.fill(1.0) }
    for key in keys {
        let valid = key.valid
        if valid is null { continue }
        reshape_kernel("frame_and_valid").dispatch_async([valid, mask.values], length, [length])
    }
    return mask
}

#: A key column reduced to dense codes, with the label of each code.
#:
#: This is the half of a reshape that decides how big the result is. `codes` is
#: one float slot per input row naming its cell along one axis, `count` is how
#: many cells that axis has, and `labels` is one column per key column holding
#: the label of each code — the thing the output index or the output column
#: names are made of.
class Factorized {
    var codes: Buffer
    var count: int
    var length: int
    var labels: list[Column]
    var names: list[str?]

    fn init(self, codes: Buffer, count: int, length: int, labels: list[Column], names: list[str?]) {
        self.codes = codes
        self.count = count
        self.length = length
        self.labels = labels
        self.names = names
    }

    fn index(self) -> Index { return Index.of(self.labels.map(|level| level.copy()), self.names) }

    fn free(self) -> void {
        self.codes.free()
        for level in self.labels { level.free() }
    }
}

#: Dense codes for one column, in first-appearance order, plus the row each
#: code was first seen at.
#:
#: An open-addressed table of row numbers on the device, then a scan. The scan
#: is what buys the ordering: a parallel hash table hands out slots in whatever
#: order the threads arrive, but the flag "this row is the first of its key" is
#: order-free, and an exclusive sum of those flags is exactly the rank of each
#: key by first appearance.
fn factorize_one(keys: Column) -> (Buffer, int, Buffer) {
    let length = keys.length
    if length == 0 { return (Buffer(0), 0, Buffer(0)) }
    require_device("factorize a key column")
    let hashes = Buffer(length)
    hashes.fill_zero()
    keys.hash_into(hashes)
    let slots = next_power_of_two(length * 2)
    let table = Buffer(slots)
    table.fill_zero()
    let slot_of_row = Buffer(length)
    reshape_kernel("frame_factorize_probe").dispatch_async(
        [
            hashes,
            keys.values,
            keys.wide ?? keys.values,
            keys.valid ?? keys.values,
            table,
            slot_of_row,
        ],
        length,
        [length, slots - 1, flag(keys.dtype.is_wide()), flag(keys.valid is not null)]
    )
    let owner = Buffer(length)
    let first = Buffer(length)
    reshape_kernel("frame_factorize_mark").dispatch_async(
        [table, slot_of_row, owner, first],
        length,
        [length]
    )
    let scanned = prefix_sum(first, length)
    let offsets = scanned[0]
    let distinct = scanned[1]
    let codes = Buffer(length)
    let first_rows = Buffer(distinct)
    reshape_kernel("frame_factorize_codes").dispatch_async(
        [owner, first, offsets, codes, first_rows],
        length,
        [length]
    )
    hashes.free()
    table.free()
    slot_of_row.free()
    owner.free()
    first.free()
    offsets.free()
    return (codes, distinct, first_rows)
}

#: Whether every key is a `Category` with no missing rows.
#:
#: The unobserved code space is the level table, and a missing row has no code
#: in it — its slot holds a zero that would be read as the first level. So a
#: nullable key falls back to being factorized, where a null is a key like any
#: other.
fn all_dense_categories(keys: list[Column]) -> bool {
    for key in keys {
        if key.dtype != Dtype.Category { return false }
        if key.has_validity() { return false }
        if key.level_count() == 0 { return false }
    }
    return keys.len() > 0
}

#: The cartesian product of the keys' level tables, in level order.
#:
#: This is what `observed: false` means: a `Category` declares its domain, so a
#: level no row uses still gets a cell. Codes are already dense — a category's
#: code lane is its level index — so there is no hash table here at all, only
#: the mixed-radix combination.
fn dense_category_keys(keys: list[Column], names: list[str?], length: int) -> Factorized {
    var cardinalities: list[int] = []
    var total = 1
    for key in keys {
        cardinalities.push(key.level_count())
        total = total * key.level_count()
    }
    if total > MAX_ROWS {
        throw ValueError(
            f"{total} unobserved category combinations exceeds the {MAX_ROWS} a grid can hold; pass observed: true"
        )
    }
    let codes = Buffer(length)
    if length > 0 { codes.fill_zero() }
    for (at, key) in keys.enumerate() {
        if length == 0 { break }
        reshape_kernel("frame_combine_dense").dispatch_async(
            [codes, key.values],
            length,
            [length, cardinalities[at]]
        )
    }
    var labels: list[Column] = []
    var inner = total
    for (at, key) in keys.enumerate() {
        inner = inner // cardinalities[at]
        let levels = Column.from_strs([name for name in key.levels ?? []], Dtype.Category)
        let positions = Buffer(total)
        if total > 0 {
            reshape_kernel("frame_unstride").dispatch_async([positions], total, [total, if inner > 0 {
                inner
            } else {
                1
            }, cardinalities[at]])
        }
        labels.push(levels.take(positions, total))
        levels.free()
        positions.free()
    }
    return Factorized(codes, total, length, labels, names)
}

#: Dense codes over one or several key columns, with a label column apiece.
#:
#: Several keys are combined rather than hashed together: each is factorized on
#: its own, then the codes are folded into one mixed-radix integer and that is
#: factorized again. The alternative is a probe kernel that compares an
#: arbitrary number of key columns, which needs a variable number of bound
#: buffers; two passes over a lane cost far less than that costs to write.
fn factorize_keys(keys: list[Column], names: list[str?], observed: bool) -> Factorized {
    if keys.len() == 0 {
        throw ValueError("a reshape needs at least one key column to group by")
    }
    let length = keys[0].length
    for key in keys {
        if key.length != length {
            throw ValueError(f"key columns disagree on length: {key.length} against {length}")
        }
    }
    if not observed and all_dense_categories(keys) {
        return dense_category_keys(keys, names, length)
    }
    if keys.len() == 1 {
        let one = factorize_one(keys[0])
        let labels = [keys[0].take(one[2], one[1])]
        one[2].free()
        return Factorized(one[0], one[1], length, labels, names)
    }
    let combined = Column.allocate(Dtype.Int64, length)
    if length > 0 {
        combined.values.fill_zero()
        (combined.wide ?? combined.values).fill_zero()
    }
    var product = 1
    for key in keys {
        let one = factorize_one(key)
        product = product * (if one[1] > 0 { one[1] } else { 1 })
        if product >= WIDE_LIMIT {
            one[0].free()
            one[2].free()
            combined.free()
            throw ValueError(
                "the key columns have too many distinct combinations to number exactly"
            )
        }
        if length > 0 and one[1] > 0 {
            reshape_kernel("frame_combine_codes").dispatch_async(
                [combined.values, combined.wide ?? combined.values, one[0]],
                length,
                [length, one[1]]
            )
        }
        one[0].free()
        one[2].free()
    }
    let dense = factorize_one(combined)
    combined.free()
    let labels = keys.map(|key| key.take(dense[2], dense[1]))
    dense[2].free()
    return Factorized(dense[0], dense[1], length, labels, names)
}

#: One source row per output cell, `MISSING_POSITION` where no row landed.
#:
#: The cells run column-major — cell `c * rows + r` — so that each output
#: column is one contiguous run and `slice_grid` can lift it out with a copy
#: instead of a strided gather.
#:
#: Raises on a duplicate pair. A reshape that silently kept the last writer
#: would turn a data error into a plausible-looking table, and the caller who
#: meant to aggregate has `pivot_table` to say so with.
fn scatter_positions(rows: Factorized, cols: Factorized, what: str) -> Buffer {
    let cells = rows.count * cols.count
    if cells > MAX_ROWS {
        throw ValueError(
            f"{rows.count} by {cols.count} is {cells} cells, past the {MAX_ROWS} a position lane addresses"
        )
    }
    let positions = filled_lane(cells, float(MISSING_POSITION))
    if rows.length == 0 or cells == 0 { return positions }
    let claim = filled_u32_lane(cells, 0)
    let conflict = Buffer(1)
    conflict.fill_zero()
    reshape_kernel("frame_pivot_scatter").dispatch_async(
        [rows.codes, cols.codes, claim, positions, conflict],
        rows.length,
        [rows.length, rows.count]
    )
    let clashed = conflict.download(0, 1)[0] != 0.0
    claim.free()
    conflict.free()
    if clashed {
        positions.free()
        throw ValueError(
            f"{what} has duplicate entries for a (row, column) pair; aggregate them instead"
        )
    }
    return positions
}

fn agg_op(agg: Agg) -> int {
    return match agg {
        Agg.Sum => AGG_SUM,
        Agg.Mean => AGG_MEAN,
        Agg.Count => AGG_COUNT,
        Agg.Size => AGG_SIZE,
        Agg.Min => AGG_MIN,
        Agg.Max => AGG_MAX,
        Agg.First => AGG_FIRST,
        Agg.Last => AGG_LAST,
        Agg.Any => AGG_ANY,
        Agg.All => AGG_ALL,
        _ => throw ValueError(f"a pivot cannot compute {agg.name()} in one scatter; group by the two keys and aggregate there"),
    }
}

#: Whether the reduction into `aux` is a minimum, and so starts from the top.
fn starts_high(op: int) -> bool {
    return op == AGG_MIN or op == AGG_FIRST or op == AGG_ALL
}

#: Whether the result is one of the input rows rather than a new number.
fn selects_a_row(op: int) -> bool {
    return op == AGG_MIN or op == AGG_MAX or op == AGG_FIRST or op == AGG_LAST
}

#: What a reduction leaves behind, given what went in.
#:
#: A reduction that picks a row keeps the input's dtype exactly, because the
#: answer is gathered from the row that won rather than recomputed. Everything
#: else reads the column's float lane, which is float32, so a sum or a mean
#: comes back as `Float64` whatever it was summing — the same compromise
#: `as_float_lane` makes and for the same reason.
fn agg_result_dtype(agg: Agg, source: Dtype) -> Dtype {
    if agg.picks_a_row() { return source }
    if agg.counts_rows() { return Dtype.Int64 }
    return match agg {
        Agg.Any | Agg.All => Dtype.Bool,
        _ => Dtype.Float64,
    }
}

#: The whole `rows` by `columns` grid of one aggregation, as one flat column.
#:
#: One dispatch over the input rows, each row folding itself into its cell with
#: an atomic. The alternative — a pass per cell — reads the data once per
#: output cell, which for a wide pivot is the same rows a hundred times over.
#:
#: `margins` widens the grid by one row and one column and has every input row
#: fold itself into four cells rather than one. Every reduction here is
#: associative, so a margin cell that simply sees more of the rows is the right
#: answer, and it costs three more atomics rather than a second pass.
fn aggregate_grid(
    rows: Factorized,
    cols: Factorized,
    value: Column?,
    agg: Agg,
    margins: bool
) -> Column {
    let op = agg_op(agg)
    let extra = if margins { 1 } else { 0 }
    let height = rows.count + extra
    let width = cols.count + extra
    let cells = height * width
    let length = rows.length
    let source = value
    if source is null and selects_a_row(op) {
        throw ValueError(f"{agg.name()} needs a values column to pick from")
    }
    if cells > MAX_ROWS {
        throw ValueError(
            f"{height} by {width} is {cells} cells, past the {MAX_ROWS} a column holds"
        )
    }
    let acc = filled_lane(cells, 0.0)
    let aux = filled_u32_lane(cells, if starts_high(op) { EMPTY_SLOT } else { 0 })
    var lane = rows.codes
    var owned = false
    var valid_lane = rows.codes
    var has_valid = false
    if source is not null {
        let borrowed = source.borrow_float_lane()
        lane = borrowed[0]
        owned = borrowed[1]
        let mask = source.valid
        if mask is not null {
            valid_lane = mask
            has_valid = true
        }
    }
    if length > 0 {
        reshape_kernel("frame_agg_cells").dispatch_async(
            [lane, valid_lane, rows.codes, cols.codes, acc, aux],
            length,
            [length, rows.count, cols.count, op, flag(has_valid), flag(margins)]
        )
    }
    if selects_a_row(op) {
        let picked = gather_selected(
            source ?? rows.labels[0],
            lane,
            valid_lane,
            rows,
            cols,
            acc,
            aux,
            op,
            has_valid,
            margins,
            cells
        )
        if owned { lane.free() }
        acc.free()
        aux.free()
        return picked
    }
    if owned { lane.free() }
    let dtype = agg_result_dtype(agg, (source ?? rows.labels[0]).dtype)
    let out = Column.allocate(dtype, cells, with_validity: true)
    if cells > 0 {
        reshape_kernel("frame_agg_finalize").dispatch_async(
            [acc, aux, out.values, out.wide ?? out.values, out.valid ?? out.values],
            cells,
            [cells, op, flag(dtype.is_wide())]
        )
    }
    acc.free()
    aux.free()
    return out
}

#: The winning row of every cell, gathered back through `take`.
#:
#: A minimum computed in the value lane is a float32 minimum, but the answer
#: this hands back is the input cell itself — so an `Int64` keeps all 48 of its
#: bits and a `Str` keeps its text, neither of which survives a trip through
#: the float lane the comparison ran in.
fn gather_selected(
    source: Column,
    lane: Buffer,
    valid_lane: Buffer,
    rows: Factorized,
    cols: Factorized,
    acc: Buffer,
    aux: Buffer,
    op: int,
    has_valid: bool,
    margins: bool,
    cells: int
) -> Column {
    if source.dtype.is_dictionary() and (op == AGG_MIN or op == AGG_MAX) {
        throw TypeError(
            f"a {source.dtype.name()} extremum is lexicographic, which needs the levels sorted; use first or last"
        )
    }
    var chosen = aux
    var owned_chosen = false
    if op == AGG_MIN or op == AGG_MAX {
        chosen = filled_u32_lane(cells, 0)
        owned_chosen = true
        if rows.length > 0 {
            reshape_kernel("frame_agg_pick_row").dispatch_async(
                [lane, valid_lane, rows.codes, cols.codes, aux, chosen],
                rows.length,
                [rows.length, rows.count, cols.count, flag(has_valid), flag(margins)]
            )
        }
    }
    let positions = Buffer(cells)
    if cells > 0 {
        reshape_kernel("frame_agg_positions").dispatch_async(
            [acc, chosen, positions],
            cells,
            [cells]
        )
    }
    let out = source.take(positions, cells, allow_missing: true)
    positions.free()
    if owned_chosen { chosen.free() }
    return out
}
