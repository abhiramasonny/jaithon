#: Products: `matmul`, `dot`, `vdot`, `inner`, `outer`, `tensordot`, `kron`,
#: `cross`, `matrix_power`, and a two-operand `einsum`.
#:
#: Everything here is one GEMM, or a short list of them, or an elementwise pass
#: that would only be a GEMM out of habit. `tensordot` is a transpose, a
#: reshape and one product; `dot` and `inner` are `tensordot` with the axes
#: numpy's rules pick; `outer` and `kron` are broadcast multiplies, because an
#: inner dimension of one gives every output a single multiply and nothing for
#: a tile to accumulate.
#:
#: **`einsum` is transpose plus reshape plus matmul, honestly and only that.**
#: Two operands, explicit subscripts, no ellipsis. The labels sort into batch,
#: free and contracted; the operands are permuted into `[batch, free,
#: contracted]` and `[batch, contracted, free]`, collapsed to three dimensions
#: and multiplied; the result is permuted into the order the arrow asked for.
#: A label in one operand and nowhere else is summed away first by contracting
#: it against a vector of ones, which is a matrix product too. There is no
#: contraction-order optimiser and no n-operand path — with two operands there
#: is no order to choose — so nobody should expect `np.einsum`'s planning.
#:
#: **The GEMM goes to Metal Performance Shaders when the operands allow it.**
#: `std.gpu.matmul_async` needs every physical matrix to have a row that is a
#: multiple of 16 bytes: the columns of `a` as stored, the columns of `b` as
#: stored, and the columns of the result, each divisible by four. MPS describes
#: a matrix by its row bytes and an `MPSNDArray` pads its innermost dimension
#: out to sixteen, so a row that does not divide is read wrong rather than
#: slowly. When the rule holds this dispatches MPSGraph; when it does not, the
#: kernels below run. Unlike jaitensor there is no autotuner and no plan cache:
#: a jainum call is not a training step repeated ten thousand times on one
#: shape, and measuring the candidates would cost more than the call.
#:
#: **A non-contiguous operand is materialised first, never read in place.** A
#: GEMM addresses an operand with one leading dimension, so the only layouts it
#: can honour are row-major and column-major — and column-major is free,
#: because it is what the transpose flags mean, which is why `transpose(a)` on
#: either side costs nothing. Anything else (a column block, a flip, a
#: broadcast inside the plane) is copied dense. A strided GEMM is the wrong
#: place to be clever: it would carry an index walk into the innermost loop,
#: and the copy that avoids it touches the operand exactly once.
#:
#: **A batched product loops over the batch encoding async dispatches**, one
#: per plane, and flushes once at the end rather than synchronising per matrix
#: — the point of a batch is that the queue stays full while the host walks it.
#: The leading axes broadcast as numpy's do, and a broadcast batch axis has
#: stride zero, so one plane is fed to several dispatches instead of copied.
#:
#: Complex operands are refused at the door: a complex product is four real
#: ones and belongs with the rest of the complex support, so `vdot`'s
#: conjugation is a no-op it never has to perform. Factorizations, solvers and
#: `norm` are `linalg.jai`; `trace` is `reduce.jai`.
#:
#: ```
#: let c = matmul(a, b)                    #: 2-d, batched, or broadcast
#: let g = matmul(transpose(a), a)         #: the transpose costs nothing
#: let t = tensordot(a, b, [[1, 2], [0, 1]])
#: let e = einsum("bij,bjk->bik", a, b)
#: ```
import std.gpu as gpu
from std.gpu import Buffer

from .array import BOOL, INT32, NDArray, OP_MUL, OP_SUB, PRELUDE
from .array import GROUP, binary_arith, broadcast_shapes, contiguous_strides
from .array import diagonal_view, element_count, fill, kernel_of, normalize_axis
from .array import normalize_axes, promote, require_real
from .manipulate import expand_dims, reshape, moveaxis, stack, transpose

#: Side of the output block one `mm_tile` threadgroup owns.
const TILE = 64

#: Outputs at or below which the fold kernel wins over the tiled one.
#:
#: 4096 is one tile: at that size `mm_tile` produces a single threadgroup and
#: the rest of the device has nothing to do, however long the inner axis is.
#: `mm_fold` takes its parallelism from that axis instead.
const FOLD_OUTPUTS = TILE * TILE

#: Threadgroups the two-pass dot product runs over at most, so that the second
#: pass folds what one group can hold: 1024 partials is four per thread.
const DOT_GROUPS = 1024

let SOURCE = PRELUDE + """
/* Tile geometry. 64x64 outputs over 256 threads is a 4x4 accumulator per
 * thread, held in four float4s: a step of the inner axis stages 64 + 64 values
 * and spends 64*64 multiplies on them, so threadgroup memory is read sixteen
 * times for each time it is written. BK of 8 keeps both stages to 4 KB
 * together, which leaves room for the accumulators. */
constexpr constant uint MM_BM = 64u;
constexpr constant uint MM_BN = 64u;
constexpr constant uint MM_BK = 8u;
constexpr constant uint MM_TM = 4u;
constexpr constant uint MM_TN = 4u;
constexpr constant uint MM_LOADS = MM_BM * MM_BK / ND_GROUP;

/* `flags` bit 0: `a` is stored column-major. Bit 1: `b` is. Bit 2: both
 * operands walk the inner axis with unit stride from a 16-byte boundary, so
 * `mm_fold` may read them four at a time. */

kernel void mm_tile(device float* out [[buffer(0)]],
                    device const float* a [[buffer(1)]],
                    device const float* b [[buffer(2)]],
                    constant uint& rows [[buffer(3)]],
                    constant uint& inner [[buffer(4)]],
                    constant uint& columns [[buffer(5)]],
                    constant uint& flags [[buffer(6)]],
                    constant uint& a_at [[buffer(7)]],
                    constant uint& b_at [[buffer(8)]],
                    constant uint& out_at [[buffer(9)]],
                    uint group [[threadgroup_position_in_grid]],
                    uint lid [[thread_index_in_threadgroup]]) {
    /* Staged as float4 so the arrays are 16-byte aligned and the inner loop
     * can take a whole row of the micro-tile in one load. */
    threadgroup float4 a_pool[MM_BM * MM_BK / 4u];
    threadgroup float4 b_pool[MM_BK * MM_BN / 4u];
    threadgroup float* As = (threadgroup float*)a_pool;
    threadgroup float* Bs = (threadgroup float*)b_pool;

    const bool trans_a = (flags & 1u) != 0u;
    const bool trans_b = (flags & 2u) != 0u;
    const uint across = (columns + MM_BN - 1u) / MM_BN;
    const uint band = group / across;
    const uint tile_row = band * MM_BM;
    const uint tile_col = (group - band * across) * MM_BN;
    const uint ty = lid / (MM_BN / MM_TN);
    const uint tx = lid - ty * (MM_BN / MM_TN);

    device const float* ap = a + a_at;
    device const float* bp = b + b_at;

    float4 acc[MM_TM];
    for (uint i = 0u; i < MM_TM; i++) acc[i] = float4(0.0f);

    for (uint k0 = 0u; k0 < inner; k0 += MM_BK) {
        /* Guards the previous pass's reads before this one overwrites them. */
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0u; t < MM_LOADS; t++) {
            const uint idx = lid + t * ND_GROUP;
            /* Neighbouring threads must walk whichever axis the operand is
             * contiguous along, or the loads stop coalescing -- so the split
             * of the flat load index goes the other way for a column-major
             * operand. One divide and a subtraction either way. */
            uint mm = idx / MM_BK;
            uint kk = idx - mm * MM_BK;
            if (trans_a) {
                kk = idx / MM_BM;
                mm = idx - kk * MM_BM;
            }
            const uint gm = tile_row + mm;
            const uint gk = k0 + kk;
            float value = 0.0f;
            if (gm < rows && gk < inner) {
                value = trans_a ? ap[gk * rows + gm] : ap[gm * inner + gk];
            }
            As[kk * MM_BM + mm] = value;
        }
        for (uint t = 0u; t < MM_LOADS; t++) {
            const uint idx = lid + t * ND_GROUP;
            uint kk = idx / MM_BN;
            uint nn = idx - kk * MM_BN;
            if (trans_b) {
                nn = idx / MM_BK;
                kk = idx - nn * MM_BK;
            }
            const uint gn = tile_col + nn;
            const uint gk = k0 + kk;
            float value = 0.0f;
            if (gn < columns && gk < inner) {
                value = trans_b ? bp[gn * inner + gk] : bp[gk * columns + gn];
            }
            Bs[kk * MM_BN + nn] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < MM_BK; kk++) {
            const float4 av = a_pool[kk * (MM_BM / 4u) + ty];
            const float4 bv = b_pool[kk * (MM_BN / 4u) + tx];
            acc[0] += av.x * bv;
            acc[1] += av.y * bv;
            acc[2] += av.z * bv;
            acc[3] += av.w * bv;
        }
    }

    /* The whole micro-tile stores as one float4 when the result's rows are a
     * multiple of four -- which is also when MPS would have taken the product,
     * so this path is for the tiles a misaligned `a` or `b` sent here. */
    const uint column = tile_col + tx * MM_TN;
    const bool packed = (columns & 3u) == 0u && column + MM_TN <= columns;
    device float* dst = out + out_at + column;
    for (uint i = 0u; i < MM_TM; i++) {
        const uint r = tile_row + ty * MM_TM + i;
        if (r >= rows) return;
        device float* line = dst + r * columns;
        if (packed) {
            *((device float4*)line) = acc[i];
            continue;
        }
        if (column < columns) line[0] = acc[i].x;
        if (column + 1u < columns) line[1] = acc[i].y;
        if (column + 2u < columns) line[2] = acc[i].z;
        if (column + 3u < columns) line[3] = acc[i].w;
    }
}

/* One threadgroup per output element, folding along the inner axis.
 *
 * `mm_tile` cuts the *output* into tiles, so a small result leaves the device
 * idle however long the product actually is: a 4x4 answer over a million-long
 * inner axis is sixteen million multiplies in one threadgroup. This is the
 * same product with the parallelism taken from the axis that has it. */
kernel void mm_fold(device float* out [[buffer(0)]],
                    device const float* a [[buffer(1)]],
                    device const float* b [[buffer(2)]],
                    constant uint& rows [[buffer(3)]],
                    constant uint& inner [[buffer(4)]],
                    constant uint& columns [[buffer(5)]],
                    constant uint& flags [[buffer(6)]],
                    constant uint& a_at [[buffer(7)]],
                    constant uint& b_at [[buffer(8)]],
                    constant uint& out_at [[buffer(9)]],
                    uint group [[threadgroup_position_in_grid]],
                    uint lid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float warp_totals[ND_WARPS];
    const uint r = group / columns;
    const uint c = group - r * columns;
    const bool trans_a = (flags & 1u) != 0u;
    const bool trans_b = (flags & 2u) != 0u;
    device const float* row = a + a_at + (trans_a ? r : r * inner);
    device const float* col = b + b_at + (trans_b ? c * inner : c);
    const uint a_step = trans_a ? rows : 1u;
    const uint b_step = trans_b ? 1u : columns;

    float total = 0.0f;
    if ((flags & 4u) != 0u) {
        device const float4* row4 = (device const float4*)row;
        device const float4* col4 = (device const float4*)col;
        const uint quads = inner / 4u;
        for (uint q = lid; q < quads; q += ND_GROUP) {
            const float4 part = row4[q] * col4[q];
            total += part.x + part.y + part.z + part.w;
        }
    } else {
        for (uint k = lid; k < inner; k += ND_GROUP) total += row[k * a_step] * col[k * b_step];
    }

    total = simd_sum(total);
    if (lane == 0u) warp_totals[warp] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0u) return;
    float sum = 0.0f;
    for (uint j = 0u; j < ND_WARPS; j++) sum += warp_totals[j];
    out[out_at + group] = sum;
}

/* Pass one of a vector dot product: each group folds its share of the inner
 * axis. One output means `mm_fold` would run a single threadgroup, so the
 * grid is fixed and every group strides the whole vector. */
kernel void mm_dot_partial(device float* partials [[buffer(0)]],
                           device const float* a [[buffer(1)]],
                           device const float* b [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant uint& packed [[buffer(4)]],
                           constant uint& a_at [[buffer(5)]],
                           constant uint& b_at [[buffer(6)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint groups [[threadgroups_per_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float warp_totals[ND_WARPS];
    const uint start = group * ND_GROUP + lid;
    const uint stride = groups * ND_GROUP;
    float total = 0.0f;
    if (packed != 0u) {
        device const float4* av = (device const float4*)(a + a_at);
        device const float4* bv = (device const float4*)(b + b_at);
        const uint quads = count / 4u;
        for (uint q = start; q < quads; q += stride) {
            const float4 part = av[q] * bv[q];
            total += part.x + part.y + part.z + part.w;
        }
    } else {
        for (uint at = start; at < count; at += stride) total += a[a_at + at] * b[b_at + at];
    }
    total = simd_sum(total);
    if (lane == 0u) warp_totals[warp] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0u) return;
    float sum = 0.0f;
    for (uint j = 0u; j < ND_WARPS; j++) sum += warp_totals[j];
    partials[group] = sum;
}

/* Pass two: one group folds the partials into the single answer. */
kernel void mm_dot_finish(device float* out [[buffer(0)]],
                          device const float* partials [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant uint& out_at [[buffer(3)]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float warp_totals[ND_WARPS];
    float total = 0.0f;
    for (uint at = lid; at < count; at += ND_GROUP) total += partials[at];
    total = simd_sum(total);
    if (lane == 0u) warp_totals[warp] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0u) return;
    float sum = 0.0f;
    for (uint j = 0u; j < ND_WARPS; j++) sum += warp_totals[j];
    out[out_at] = sum;
}
"""

#: The dtype a product carries. Two booleans multiply and *sum*, which is
#: counting, so the answer is an integer rather than another boolean.
fn _product_dtype(a: int, b: int) -> int {
    let out = promote(a, b)
    if out == BOOL { return INT32 }
    return out
}

#: Whether MPS will take this shape, which is entirely a question of row
#: length.
#:
#: The columns of each matrix *as it is stored* must be a multiple of four
#: floats — a transposed operand is stored with the other extent innermost, so
#: the flag decides which of the two is checked.
fn _mps_ready(rows: int, inner: int, columns: int, trans_a: bool, trans_b: bool) -> bool {
    let a_columns = if trans_a { rows } else { inner }
    let b_columns = if trans_b { inner } else { columns }
    return a_columns % 4 == 0 and b_columns % 4 == 0 and columns % 4 == 0
}

#: `a` with its trailing two axes in a layout a GEMM can address: the array
#: itself when they are row-major or column-major, a dense copy otherwise.
#:
#: The middle flag says the plane is stored transposed, which costs nothing —
#: it is what MPS's `trans_a` and the kernels' flag bits are for. The last says
#: the copy happened and the caller owns what came back.
#:
#: An axis of one element carries a stride nothing reads, so a stale one must
#: not send a row vector through a copy it does not need.
fn _plane(a: NDArray) -> (NDArray, bool, bool) {
    let rank = a.ndim()
    let rows = a.shape[rank - 2]
    let columns = a.shape[rank - 1]
    let down = a.strides[rank - 2]
    let across = a.strides[rank - 1]
    if (columns == 1 or across == 1) and (rows == 1 or down == columns) {
        return (a, false, false)
    }
    if (rows == 1 or down == 1) and (columns == 1 or across == rows) {
        return (a, true, false)
    }
    return (a.copy(), false, true)
}

#: The buffer a GEMM should be handed for a plane starting at `at`.
#:
#: MPS takes buffers and not offsets, so a plane past the first travels as a
#: view. The flag says the view was allocated and has to be released after
#: encoding — an in-flight command buffer retains what it read, so releasing it
#: then is safe and holding it longer would leak a handle per plane.
fn _window(store: Buffer, at: int, count: int) -> (Buffer, bool) {
    if at == 0 { return (store, false) }
    return (store.view(at, count), true)
}

#: A vector dot product: partials over a fixed grid, then one group folds them.
fn _encode_dot(
    out: NDArray,
    out_at: int,
    a: NDArray,
    a_at: int,
    b: NDArray,
    b_at: int,
    inner: int
) -> void {
    let groups = min(DOT_GROUPS, (inner + GROUP - 1) // GROUP)
    let packed = if inner % 4 == 0 and a_at % 4 == 0 and b_at % 4 == 0 { 1 } else { 0 }
    let partials = Buffer(groups)
    kernel_of(SOURCE, "mm_dot_partial").dispatch_async(
        [partials, a.storage, b.storage],
        groups * GROUP,
        [inner, packed, a_at, b_at],
        GROUP
    )
    kernel_of(SOURCE, "mm_dot_finish").dispatch_async(
        [out.storage, partials],
        GROUP,
        [groups, out_at],
        GROUP
    )
    partials.free()
}

#: One `rows` by `columns` product, encoded and not waited on.
#:
#: The four routes in the order they are tried: a single output is a reduction
#: and nothing else; MPS takes whatever its row rule allows; a result small
#: enough to fit one tile folds along the inner axis instead; everything else
#: is the tiled kernel.
fn _encode(
    out: NDArray,
    out_at: int,
    a: NDArray,
    a_at: int,
    trans_a: bool,
    b: NDArray,
    b_at: int,
    trans_b: bool,
    rows: int,
    inner: int,
    columns: int
) -> void {
    if rows == 1 and columns == 1 {
        _encode_dot(out, out_at, a, a_at, b, b_at, inner)
        return
    }
    if _mps_ready(rows, inner, columns, trans_a, trans_b) {
        let (left, owns_left) = _window(a.storage, a_at, rows * inner)
        let (right, owns_right) = _window(b.storage, b_at, inner * columns)
        let (result, owns_result) = _window(out.storage, out_at, rows * columns)
        gpu.matmul_async(
            left,
            right,
            result,
            rows,
            inner,
            columns,
            trans_a: trans_a,
            trans_b: trans_b
        )
        if owns_left { left.free() }
        if owns_right { right.free() }
        if owns_result { result.free() }
        return
    }
    var flags = 0
    if trans_a { flags += 1 }
    if trans_b { flags += 2 }
    if rows * columns <= FOLD_OUTPUTS {
        let a_step = if trans_a { rows } else { 1 }
        let b_step = if trans_b { 1 } else { columns }
        if a_step == 1 and b_step == 1 and inner % 4 == 0 and a_at % 4 == 0 and b_at % 4 == 0 {
            flags += 4
        }
        kernel_of(SOURCE, "mm_fold").dispatch_async(
            [out.storage, a.storage, b.storage],
            rows * columns * GROUP,
            [rows, inner, columns, flags, a_at, b_at, out_at],
            GROUP
        )
        return
    }
    let across = (columns + TILE - 1) // TILE
    let down = (rows + TILE - 1) // TILE
    kernel_of(SOURCE, "mm_tile").dispatch_async(
        [out.storage, a.storage, b.storage],
        across * down * GROUP,
        [rows, inner, columns, flags, a_at, b_at, out_at],
        GROUP
    )
}

#: Every plane of a batched product, one dispatch each and one flush at the end.
#:
#: The offsets are carried by an odometer over the batch axes rather than
#: recomputed per plane, so a broadcast axis — stride zero — feeds the same
#: plane to as many dispatches as it stretches to and copies nothing.
fn _product(
    out: NDArray,
    a: NDArray,
    trans_a: bool,
    b: NDArray,
    trans_b: bool,
    batch: list[int],
    rows: int,
    inner: int,
    columns: int
) -> void {
    let planes = element_count(batch)
    if planes == 0 or rows == 0 or columns == 0 { return }
    if inner == 0 {
        fill(out, 0.0)
        return
    }
    let rank = batch.len()
    var counter = [0 for _axis in batch]
    var a_at = a.offset
    var b_at = b.offset
    var out_at = out.offset
    for _plane in 0..planes {
        _encode(out, out_at, a, a_at, trans_a, b, b_at, trans_b, rows, inner, columns)
        var axis = rank - 1
        while axis >= 0 {
            counter[axis] += 1
            a_at += a.strides[axis]
            b_at += b.strides[axis]
            out_at += out.strides[axis]
            if counter[axis] < batch[axis] { break }
            a_at -= batch[axis] * a.strides[axis]
            b_at -= batch[axis] * b.strides[axis]
            out_at -= batch[axis] * out.strides[axis]
            counter[axis] = 0
            axis -= 1
        }
    }
    if planes > 1 { gpu.flush() }
}

#: The stacked product of two arrays of at least two dimensions, with the
#: leading axes broadcast against each other.
fn _stacked(a: NDArray, b: NDArray) -> NDArray {
    require_real(a, "a matrix product")
    require_real(b, "a matrix product")
    let ra = a.ndim()
    let rb = b.ndim()
    let rows = a.shape[ra - 2]
    let inner = a.shape[ra - 1]
    let shared = b.shape[rb - 2]
    let columns = b.shape[rb - 1]
    if inner != shared {
        throw ValueError(f"cannot multiply {a.shape} by {b.shape}: {inner} against {shared}")
    }
    let batch = broadcast_shapes(a.shape[0:ra - 2], b.shape[0:rb - 2])
    let (a_base, trans_a, owns_a) = _plane(a)
    let (b_base, trans_b, owns_b) = _plane(b)
    let out = NDArray(batch + [rows, columns], _product_dtype(a.dtype, b.dtype))
    _product(
        out,
        a_base.broadcast_to(batch + [rows, inner]),
        trans_a,
        b_base.broadcast_to(batch + [inner, columns]),
        trans_b,
        batch,
        rows,
        inner,
        columns
    )
    if owns_a { a_base.free() }
    if owns_b { b_base.free() }
    return out
}

#: `a` seen as `shape`, and whether that cost an allocation the caller owns.
#:
#: `reshape` hands back the array itself when the shape already matches and a
#: view when the strides allow one, and a view's `free` is a no-op; only the
#: materialised case has an owner, and it is the one whose `base` is null.
fn _recast(a: NDArray, shape: list[int]) -> (NDArray, bool) {
    if shape == a.shape { return (a, false) }
    let out = reshape(a, shape)
    return (out, out.base is null)
}

#: A freshly produced array relabelled with `shape`, still owning its storage.
#:
#: Not `reshape`: that returns a *view*, whose `free` does nothing because a
#: view does not own an allocation, so handing one back as a result would leak
#: the product it is a window onto. Every entry point here that reshapes its
#: own answer goes through this.
fn _relabel(a: NDArray, shape: list[int]) -> NDArray {
    if element_count(shape) != a.size() {
        throw ValueError(f"cannot relabel {a.shape} as {shape}")
    }
    return NDArray(shape, a.dtype, a.storage, contiguous_strides(shape, a.lanes()), a.offset)
}

#: One component of the last axis, as a view with that axis dropped.
fn _component(a: NDArray, at: int) -> NDArray {
    let rank = a.ndim()
    return a.view(a.shape[0:rank - 1], a.strides[0:rank - 1], a.offset + at * a.strides[rank - 1])
}

#: `a * b - c * d`, elementwise, freeing the two products.
fn _cross_term(a: NDArray, b: NDArray, c: NDArray, d: NDArray) -> NDArray {
    let left = binary_arith(a, b, OP_MUL)
    let right = binary_arith(c, d, OP_MUL)
    let out = binary_arith(left, right, OP_SUB)
    left.free()
    right.free()
    return out
}

#: numpy's `matmul`, and the `@` this language does not spell.
#:
#: Two matrices multiply; anything with more axes is a stack of matrices whose
#: leading axes broadcast. A 1-d operand is treated as a row on the left or a
#: column on the right and the axis that promoted it is dropped again from the
#: result — which is why `matmul` of two vectors is a scalar and `dot` of a
#: matrix and a vector is a vector. A 0-d operand has no reading at all here,
#: as in numpy, because scaling is not a product of matrices.
pub fn matmul(a: NDArray, b: NDArray) -> NDArray {
    if a.ndim() == 0 or b.ndim() == 0 {
        throw ValueError("matmul has no reading for a 0-d operand; multiply() scales")
    }
    let left = if a.ndim() == 1 { expand_dims(a, 0) } else { a }
    let right = if b.ndim() == 1 { expand_dims(b, 1) } else { b }
    let product = _stacked(left, right)
    if a.ndim() > 1 and b.ndim() > 1 { return product }
    let rank = product.ndim()
    var shape: list[int] = []
    for axis in 0..rank {
        if a.ndim() == 1 and axis == rank - 2 { continue }
        if b.ndim() == 1 and axis == rank - 1 { continue }
        shape.push(product.shape[axis])
    }
    return _relabel(product, shape)
}

#: Two vectors' inner product, as a 0-d array.
fn _dot1d(a: NDArray, b: NDArray) -> NDArray {
    return _relabel(_stacked(expand_dims(a, 0), expand_dims(b, 1)), [])
}

#: numpy's `dot`, which is five operations wearing one name.
#:
#: A 0-d operand scales; two vectors give their inner product as a 0-d array;
#: two matrices multiply; and in general the last axis of `a` is contracted
#: against the second-to-last of `b` — the last axis of `b` when `b` is a
#: vector. That last rule is what makes `dot` differ from `matmul` on stacked
#: input: `dot` takes the outer product of the two batches where `matmul`
#: broadcasts them together.
pub fn dot(a: NDArray, b: NDArray) -> NDArray {
    require_real(a, "dot")
    require_real(b, "dot")
    if a.ndim() == 0 or b.ndim() == 0 { return binary_arith(a, b, OP_MUL) }
    if a.ndim() == 1 and b.ndim() == 1 { return _dot1d(a, b) }
    if a.ndim() == 2 and b.ndim() == 2 { return _stacked(a, b) }
    if b.ndim() == 1 { return tensordot(a, b, [[a.ndim() - 1], [0]]) }
    return tensordot(a, b, [[a.ndim() - 1], [b.ndim() - 2]])
}

#: Both operands flattened, then their inner product, as a 0-d array.
#:
#: numpy conjugates the first argument. Complex is refused here, so there is
#: nothing to conjugate and this differs from `dot` only in the flattening.
pub fn vdot(a: NDArray, b: NDArray) -> NDArray {
    require_real(a, "vdot")
    require_real(b, "vdot")
    if a.size() != b.size() {
        throw ValueError(f"vdot needs matching sizes, got {a.size()} and {b.size()}")
    }
    let (x, owns_x) = _recast(a, [a.size()])
    let (y, owns_y) = _recast(b, [b.size()])
    let out = _dot1d(x, y)
    if owns_x { x.free() }
    if owns_y { y.free() }
    return out
}

#: The sum of products over the *last* axis of both operands — numpy's
#: `inner`, which is `dot` with `b` transposed.
pub fn inner(a: NDArray, b: NDArray) -> NDArray {
    require_real(a, "inner")
    require_real(b, "inner")
    if a.ndim() == 0 or b.ndim() == 0 { return binary_arith(a, b, OP_MUL) }
    return tensordot(a, b, [[a.ndim() - 1], [b.ndim() - 1]])
}

#: Every product of an element of `a` with an element of `b`, both flattened.
#:
#: A broadcast multiply and not a GEMM with an inner dimension of one: every
#: output is one multiply, and there is nothing for a tile to accumulate.
pub fn outer(a: NDArray, b: NDArray) -> NDArray {
    require_real(a, "outer")
    require_real(b, "outer")
    let (column, owns_column) = _recast(a, [a.size(), 1])
    let (row, owns_row) = _recast(b, [1, b.size()])
    let out = binary_arith(column, row, OP_MUL)
    if owns_column { column.free() }
    if owns_row { row.free() }
    return out
}

#: The axes `tensordot` contracts, from a count or from an explicit pair.
fn _axis_pair(axes: any, ra: int, rb: int) -> (list[int], list[int]) {
    if axes is int {
        if axes < 0 { throw ValueError(f"tensordot takes a count of axes, not {axes}") }
        if axes > ra or axes > rb {
            throw ValueError(f"cannot contract {axes} axes of arrays of {ra} and {rb} dimensions")
        }
        var left: list[int] = []
        var right: list[int] = []
        for at in 0..axes {
            left.push(ra - axes + at)
            right.push(at)
        }
        return (left, right)
    }
    var pair: list[any] = []
    if axes is tuple or axes is list {
        for item in axes { pair.push(item) }
    } else {
        throw TypeError(f"tensordot axes is a count or a pair of axis lists, not {type_of(axes)}")
    }
    if pair.len() != 2 {
        throw ValueError(f"tensordot axes is a pair, and this one has {pair.len()} parts")
    }
    return (_axis_list(pair[0], ra), _axis_list(pair[1], rb))
}

#: One side of an axis pair: a single axis or several.
fn _axis_list(value: any, rank: int) -> list[int] {
    if value is int { return [normalize_axis(value, rank, "axes")] }
    var out: list[int] = []
    if value is tuple or value is list {
        for item in value { out.push(int(item)) }
        return normalize_axes(out, rank, "axes")
    }
    throw TypeError(f"an axis list may not be a {type_of(value)}")
}

#: The sum of products over `axes` of `a` against `axes` of `b`.
#:
#: The whole operation is a permutation and a reshape away from a matrix
#: product: the contracted axes are gathered at the inner end of `a` and the
#: outer end of `b`, both are collapsed to two dimensions, and one GEMM does
#: the arithmetic. `axes` is a count — the last that many of `a` against the
#: first that many of `b` — or a pair of axis lists.
pub fn tensordot(a: NDArray, b: NDArray, axes: any = 2) -> NDArray {
    require_real(a, "tensordot")
    require_real(b, "tensordot")
    let (left, right) = _axis_pair(axes, a.ndim(), b.ndim())
    if left.len() != right.len() {
        throw ValueError(f"{left.len()} axes of one array against {right.len()} of the other")
    }
    for at in 0..left.len() {
        if a.shape[left[at]] != b.shape[right[at]] {
            throw ValueError(
                f"axis {left[at]} of {a.shape} does not match axis {right[at]} of {b.shape}"
            )
        }
    }
    var free_left: list[int] = []
    for axis in 0..a.ndim() {
        if not left.contains(axis) { free_left.push(axis) }
    }
    var free_right: list[int] = []
    for axis in 0..b.ndim() {
        if not right.contains(axis) { free_right.push(axis) }
    }

    var shape: list[int] = []
    var rows = 1
    for axis in free_left {
        rows *= a.shape[axis]
        shape.push(a.shape[axis])
    }
    var shared = 1
    for axis in left { shared *= a.shape[axis] }
    var columns = 1
    for axis in free_right {
        columns *= b.shape[axis]
        shape.push(b.shape[axis])
    }

    let (flat_a, owns_a) = _recast(transpose(a, free_left + left), [rows, shared])
    let (flat_b, owns_b) = _recast(transpose(b, right + free_right), [shared, columns])
    let product = _stacked(flat_a, flat_b)
    if owns_a { flat_a.free() }
    if owns_b { flat_b.free() }
    return _relabel(product, shape)
}

#: The Kronecker product: every block of `a[i] * b`, laid out so that axis `k`
#: of the result runs over `a`'s axis `k` outside `b`'s.
#:
#: One broadcast multiply. Interleaving a length-one axis after each of `a`'s
#: and before each of `b`'s puts each element of `a` against the whole of `b`,
#: and the result is that array read with the pairs merged.
pub fn kron(a: NDArray, b: NDArray) -> NDArray {
    require_real(a, "kron")
    require_real(b, "kron")
    let rank = max(a.ndim(), b.ndim())
    var a_shape: list[int] = []
    var b_shape: list[int] = []
    var shape: list[int] = []
    for axis in 0..rank {
        let from_a = if axis < rank - a.ndim() { 1 } else { a.shape[axis - rank + a.ndim()] }
        let from_b = if axis < rank - b.ndim() { 1 } else { b.shape[axis - rank + b.ndim()] }
        a_shape.push(from_a)
        a_shape.push(1)
        b_shape.push(1)
        b_shape.push(from_b)
        shape.push(from_a * from_b)
    }
    let (left, owns_left) = _recast(a, a_shape)
    let (right, owns_right) = _recast(b, b_shape)
    let product = binary_arith(left, right, OP_MUL)
    if owns_left { left.free() }
    if owns_right { right.free() }
    return _relabel(product, shape)
}

#: The cross product along one axis of each operand.
#:
#: Two components or three, in any mix. numpy's one special case is two
#: against two, whose product has only the component normal to the plane and
#: so comes back without an axis to hold it; every other mix gives three
#: components, with the missing one read as zero. `axis` sets all three of
#: `axisa`, `axisb` and `axisc` at once.
#:
#: Nine elementwise dispatches and a stack, not a kernel of its own: three
#: components of two products and a difference each, all of them already
#: broadcasting passes that `array.jai` owns.
pub fn cross(
    a: NDArray,
    b: NDArray,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int? = null
) -> NDArray {
    require_real(a, "cross")
    require_real(b, "cross")
    let left = moveaxis(a, axis ?? axisa, -1)
    let right = moveaxis(b, axis ?? axisb, -1)
    let na = left.shape[left.ndim() - 1]
    let nb = right.shape[right.ndim() - 1]
    if na != 2 and na != 3 or nb != 2 and nb != 3 {
        throw ValueError(f"cross takes vectors of 2 or 3 components, not {na} and {nb}")
    }
    let batch = broadcast_shapes(left.shape[0:left.ndim() - 1], right.shape[0:right.ndim() - 1])
    let lv = left.broadcast_to(batch + [na])
    let rv = right.broadcast_to(batch + [nb])
    if na == 2 and nb == 2 {
        return _cross_term(
            _component(lv, 0),
            _component(rv, 1),
            _component(lv, 1),
            _component(rv, 0)
        )
    }
    let zeros = NDArray(batch, _product_dtype(a.dtype, b.dtype))
    fill(zeros, 0.0)
    let a2 = if na == 3 { _component(lv, 2) } else { zeros }
    let b2 = if nb == 3 { _component(rv, 2) } else { zeros }
    let parts = [
        _cross_term(_component(lv, 1), b2, a2, _component(rv, 1)),
        _cross_term(a2, _component(rv, 0), _component(lv, 0), b2),
        _cross_term(
            _component(lv, 0),
            _component(rv, 1),
            _component(lv, 1),
            _component(rv, 0)
        ),
    ]
    let out = stack(parts, axis ?? axisc)
    for part in parts { part.free() }
    zeros.free()
    return out
}

#: A batched identity shaped like `a`.
fn _identity_like(a: NDArray) -> NDArray {
    let rank = a.ndim()
    let out = NDArray(a.shape, _product_dtype(a.dtype, a.dtype))
    fill(out, 0.0)
    fill(diagonal_view(out, 0, rank - 2, rank - 1), 1.0)
    return out
}

#: `a` multiplied by itself `n` times, squaring rather than iterating.
#:
#: Binary exponentiation: about `2*log2(n)` products instead of `n - 1` of
#: them. The general path starts from the identity, which costs one product
#: that multiplies by one, so the two exponents where that would be the whole
#: cost answer directly. A negative exponent is an inverse and belongs to
#: `linalg`, which this module may not import.
pub fn matrix_power(a: NDArray, n: int) -> NDArray {
    require_real(a, "matrix_power")
    let rank = a.ndim()
    if rank < 2 { throw ValueError(f"matrix_power needs a matrix, not {a.shape}") }
    if a.shape[rank - 2] != a.shape[rank - 1] {
        throw ValueError(f"matrix_power needs square matrices, not {a.shape}")
    }
    if n < 0 {
        throw ValueError("matrix_power with a negative exponent needs linalg.inv first")
    }
    if n == 0 { return _identity_like(a) }
    if n == 1 { return a.copy() }
    if n == 2 { return _stacked(a, a) }
    var result = _identity_like(a)
    var base = a.copy()
    var power = n
    while power > 0 {
        if power % 2 == 1 {
            let next = _stacked(result, base)
            result.free()
            result = next
        }
        power = power // 2
        if power > 0 {
            let squared = _stacked(base, base)
            base.free()
            base = squared
        }
    }
    base.free()
    return result
}

#: One operand's subscripts, one label per axis.
fn _labels(text: str, rank: int, which: str) -> list[str] {
    let term = text.strip()
    if term.contains(".") {
        throw ValueError(f"einsum here has no ellipsis; name every axis of {which}")
    }
    var out: list[str] = []
    for at in 0..term.len() {
        let label = term[at]
        if out.contains(label) {
            throw ValueError(f"{which} repeats the label {label}; a diagonal is not a product")
        }
        out.push(label)
    }
    if rank >= 0 and out.len() != rank {
        throw ValueError(f"{which} has {rank} axes and {out.len()} subscripts")
    }
    return out
}

#: Record a label's extent, rejecting a second and different one.
fn _record(extent: dict[str, int], label: str, length: int) -> void {
    let known = extent.get(label, -1)
    if known >= 0 and known != length {
        throw ValueError(
            f"the label {label} is {known} long in one operand and {length} in the other"
        )
    }
    extent[label] = length
}

#: `axes` summed away, by contracting them against a vector of ones.
#:
#: A label that appears in one operand and in nothing else is a plain sum, and
#: a sum along an axis *is* a matrix product with a ones vector — so this is
#: one more GEMM rather than a reduction kernel, and this module keeps its
#: imports at `array` and `manipulate`.
fn _sum_axes(x: NDArray, axes: list[int]) -> (NDArray, bool) {
    if axes.len() == 0 { return (x, false) }
    var shape: list[int] = []
    for at in axes { shape.push(x.shape[at]) }
    let ones = NDArray(shape, x.dtype)
    fill(ones, 1.0)
    let folded = tensordot(x, ones, [axes, [at for at in 0..axes.len()]])
    ones.free()
    return (folded, true)
}

#: Where `label` sits in `labels`.
fn _position(labels: list[str], label: str) -> int {
    for at in 0..labels.len() {
        if labels[at] == label { return at }
    }
    throw ValueError(f"the label {label} is not in {labels}")
}

#: The labels left after the lone ones are summed away, in their old order.
fn _kept(labels: list[str], other: list[str], result: list[str]) -> list[str] {
    var out: list[str] = []
    for label in labels {
        if other.contains(label) or result.contains(label) { out.push(label) }
    }
    return out
}

#: The axes of `labels` that appear in neither `other` nor `result`.
fn _lone(labels: list[str], other: list[str], result: list[str]) -> list[int] {
    var out: list[int] = []
    for at in 0..labels.len() {
        if other.contains(labels[at]) or result.contains(labels[at]) { continue }
        out.push(at)
    }
    return out
}

#: A two-operand contraction in explicit subscripts: `"bij,bjk->bik"`.
#:
#: Every axis is named, both terms and the result, and the arrow is required —
#: there is no implicit output ordering to infer and no ellipsis. A label
#: shared by both operands is contracted when the result does not mention it
#: and batched when it does; a label in one operand and in the result is a free
#: axis; a label in one operand and nowhere else is summed away first.
#:
#: The work is a transpose, a reshape and one `matmul`, and the result is
#: permuted into the requested order — which is a copy, because a view of an
#: intermediate would have no owner to free it. This is not `np.einsum`: there
#: is no contraction-order optimiser here, because with two operands there is
#: no order to choose.
pub fn einsum(subscripts: str, a: NDArray, b: NDArray) -> NDArray {
    require_real(a, "einsum")
    require_real(b, "einsum")
    let sides = subscripts.split("->")
    if sides.len() != 2 {
        throw ValueError(f"einsum here needs explicit subscripts with an arrow, not {subscripts}")
    }
    let terms = sides[0].split(",")
    if terms.len() != 2 {
        throw ValueError(f"einsum here takes exactly two operands, not {terms.len()}")
    }
    let left = _labels(terms[0], a.ndim(), "the first operand")
    let right = _labels(terms[1], b.ndim(), "the second operand")
    let result = _labels(sides[1], -1, "the result")

    var extent: dict[str, int] = {}
    for at in 0..left.len() { _record(extent, left[at], a.shape[at]) }
    for at in 0..right.len() { _record(extent, right[at], b.shape[at]) }
    for label in result {
        if not left.contains(label) and not right.contains(label) {
            throw ValueError(f"the result label {label} appears in neither operand")
        }
    }

    let (ax, owns_ax) = _sum_axes(a, _lone(left, right, result))
    let (bx, owns_bx) = _sum_axes(b, _lone(right, left, result))
    let la = _kept(left, right, result)
    let lb = _kept(right, left, result)

    var batch: list[str] = []
    var contract: list[str] = []
    var free_left: list[str] = []
    var free_right: list[str] = []
    for label in la {
        if not lb.contains(label) {
            free_left.push(label)
        } elif result.contains(label) {
            batch.push(label)
        } else {
            contract.push(label)
        }
    }
    for label in lb {
        if not la.contains(label) { free_right.push(label) }
    }

    var planes = 1
    for label in batch { planes *= extent[label] }
    var rows = 1
    for label in free_left { rows *= extent[label] }
    var shared = 1
    for label in contract { shared *= extent[label] }
    var columns = 1
    for label in free_right { columns *= extent[label] }

    var order_a: list[int] = []
    for label in batch + free_left + contract { order_a.push(_position(la, label)) }
    var order_b: list[int] = []
    for label in batch + contract + free_right { order_b.push(_position(lb, label)) }

    let (flat_a, owns_a) = _recast(transpose(ax, order_a), [planes, rows, shared])
    let (flat_b, owns_b) = _recast(transpose(bx, order_b), [planes, shared, columns])
    let product = _stacked(flat_a, flat_b)
    if owns_a { flat_a.free() }
    if owns_b { flat_b.free() }
    if owns_ax { ax.free() }
    if owns_bx { bx.free() }

    let natural = batch + free_left + free_right
    var shape: list[int] = []
    for label in natural { shape.push(extent[label]) }
    let shaped = _relabel(product, shape)
    if natural == result { return shaped }
    var order: list[int] = []
    for label in result { order.push(_position(natural, label)) }
    let out = transpose(shaped, order).copy()
    shaped.free()
    return out
}
