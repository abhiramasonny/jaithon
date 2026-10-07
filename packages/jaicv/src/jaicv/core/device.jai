#: The products, scatters and projections jaicv's machine learning hands to
#: the device, through jaitensor's GEMM.
#:
#: Everything here works in float32, so it is for callers whose answer is
#: float32 anyway -- a covariance, a projection, a gradient -- and never for
#: geometry's normal equations, which `core.linalg` sums on the host in
#: double. The module is imported inside the functions that use it, so that
#: importing `core.linalg` does not load jaitensor.
#:
#: No product here is ever timed by the GEMM's tuner. The tuner measures every
#: candidate the first time it sees a shape, which costs seconds, and a
#: one-off product over a dataset is a new shape for every sample count: a
#: first principal component analysis of 77,777 samples took 9.3 s, nearly
#: all of it tuning, against 0.3 s for the host loop it replaced. So a long
#: product is cut into pieces of one fixed shape, each too small to be tuned,
#: and all of them go out in one batched dispatch.
import jaitensor as jt
from jaitensor import Tensor
from jaitensor.ops.matmul import raw_matmul_batched
from ..core.mat import Mat
from ..core.types import CV_32F, CV_MAKETYPE

#: Multiply-adds in one piece of a product. Half the size at which jaitensor's
#: GEMM starts tuning a shape (4,194,304), so that no piece is ever tuned.
pub let PIECE_WORK = 1 << 21

#: The fewest rows, or the shortest stretch of the inner dimension, worth
#: making a piece of. A product too wide to cut that finely goes to the GEMM
#: whole, and is tuned the first time its shape is seen.
let PIECE_LEAST = 64

#: A tensor reading `mat`'s values where they lie, or null when they are not
#: one unbroken run of single-channel values. Freeing the tensor releases the
#: view, not the pixels.
pub fn device_view(mat: Mat) -> Tensor? {
    if mat.channels() != 1 or not mat.is_continuous() or mat.rows * mat.cols == 0 { return null }
    let storage = mat.buffer()
    return Tensor([mat.rows, mat.cols], storage: storage.view(mat.offset, mat.rows * mat.cols))
}

#: A float matrix holding `product`'s values, which it takes over.
fn mat_of(product: Tensor) -> Mat {
    return Mat(
        product.shape[0],
        product.shape[1],
        CV_MAKETYPE(CV_32F, 1),
        storage: product.buffer()
    )
}

#: `rows` rows of `tensor`'s `width`-wide rows, from row `first`, read in place.
fn rows_of(tensor: Tensor, first: int, rows: int, width: int) -> Tensor {
    return Tensor([rows, width], storage: tensor.buffer().view(first * width, rows * width))
}

#: `a` (or its transpose) times `b` (or its transpose), without ever tuning.
#:
#: A product with many rows is cut into blocks of rows that share `b`; one
#: whose inner dimension is long and runs down both operands' rows -- a Gram
#: matrix, or a gradient summed over every sample -- is cut along that
#: dimension, and the blocks' products summed. Either way the blocks have one
#: fixed shape whatever the size of the dataset, go out as one batched
#: dispatch, and leave at most one short block over.
pub fn product(a: Tensor, b: Tensor, trans_a: bool = false, trans_b: bool = false) -> Tensor {
    let rows = if trans_a { a.shape[1] } else { a.shape[0] }
    let inner = if trans_a { a.shape[0] } else { a.shape[1] }
    let columns = if trans_b { b.shape[0] } else { b.shape[1] }
    if rows * inner * columns > PIECE_WORK {
        if not trans_a and inner * columns * PIECE_LEAST <= PIECE_WORK {
            return by_rows(a, b, trans_b, rows, inner, columns)
        }
        if trans_a and not trans_b and rows * columns * PIECE_LEAST <= PIECE_WORK {
            return by_inner(a, b, rows, inner, columns)
        }
    }
    return jt.matmul(a, b, trans_a: trans_a, trans_b: trans_b)
}

#: `a` times `b`, a block of `a`'s rows at a time.
fn by_rows(a: Tensor, b: Tensor, trans_b: bool, rows: int, inner: int, columns: int) -> Tensor {
    let piece = PIECE_WORK // (inner * columns)
    let whole = rows // piece
    let rest = rows - whole * piece
    let out = Tensor([rows, columns])
    if whole > 0 {
        let left = rows_of(a, 0, whole * piece, inner)
        let target = rows_of(out, 0, whole * piece, columns)
        raw_matmul_batched(
            left,
            b,
            target,
            whole,
            piece,
            inner,
            columns,
            piece * inner,
            0,
            trans_b: trans_b
        )
        left.free()
        target.free()
    }
    if rest > 0 {
        let left = rows_of(a, whole * piece, rest, inner)
        let target = rows_of(out, whole * piece, rest, columns)
        jt.matmul(left, b, trans_b: trans_b, out: target)
        left.free()
        target.free()
    }
    return out
}

#: `a`'s transpose times `b`, a stretch of their shared rows at a time.
#:
#: Each stretch's product lands in its own slot, and the slots are summed
#: column by column at the end, which also sums in shorter runs than one long
#: product would.
fn by_inner(a: Tensor, b: Tensor, rows: int, inner: int, columns: int) -> Tensor {
    let piece = PIECE_WORK // (rows * columns)
    let whole = inner // piece
    let rest = inner - whole * piece
    let slots = whole + (if rest > 0 { 1 } else { 0 })
    let partials = Tensor([slots * rows, columns])
    if whole > 0 {
        let left = rows_of(a, 0, whole * piece, rows)
        let right = rows_of(b, 0, whole * piece, columns)
        let target = rows_of(partials, 0, whole * rows, columns)
        raw_matmul_batched(
            left,
            right,
            target,
            whole,
            rows,
            piece,
            columns,
            piece * rows,
            piece * columns,
            trans_a: true
        )
        left.free()
        right.free()
        target.free()
    }
    if rest > 0 {
        let left = rows_of(a, whole * piece, rest, rows)
        let right = rows_of(b, whole * piece, rest, columns)
        let target = rows_of(partials, whole * rows, rows, columns)
        jt.matmul(left, right, trans_a: true, out: target)
        left.free()
        right.free()
        target.free()
    }
    if slots == 1 { return partials }
    let flat = jt.reshape(partials, [slots, rows * columns])
    let turned = jt.transpose(flat)
    flat.free()
    partials.free()
    let summed = jt.sum(turned, axis: -1)
    turned.free()
    #: A reshape is a view that freeing `summed` would pull out from under;
    #: this takes the buffer over instead.
    return Tensor([rows, columns], storage: summed.buffer())
}

#: `a` (or its transpose) times `b` (or its transpose) on the device, or null
#: when either is not a plain run of floats.
pub fn device_product(a: Mat, b: Mat, trans_a: bool = false, trans_b: bool = false) -> Mat? {
    let left = device_view(a)
    if left is null { return null }
    let right = device_view(b)
    if right is null {
        left.free()
        return null
    }
    let found = product(left, right, trans_a: trans_a, trans_b: trans_b)
    left.free()
    right.free()
    return mat_of(found)
}

#: The mean of the rows of `data` and their scatter about it -- the
#: covariance before it is divided by a count -- formed on the device: the
#: column totals by a reduction, then the centred rows' Gram matrix by a
#: GEMM. Null when `data` is not a plain run of floats.
#:
#: The device works in float32 where the host loop it replaces worked in
#: double. The scatter comes back exactly symmetric, as the eigen solver
#: expects.
pub fn device_scatter(data: Mat) -> tuple[list[float], list[float]]? {
    let rows = data.rows
    let width = data.cols
    let x = device_view(data)
    if x is null { return null }
    let turned = jt.transpose(x)
    let totals = jt.sum(turned, axis: -1)
    turned.free()
    let sums = totals.download()
    totals.free()
    var centre: list[float] = []
    var shift: list[float] = []
    for column in 0..width {
        let value = sums[column] / float(rows)
        centre.push(value)
        shift.push(-value)
    }
    let offset = Tensor([width], values: shift)
    let centred = jt.add(x, offset)
    offset.free()
    x.free()
    let gram = product(centred, centred, trans_a: true)
    centred.free()
    let raw = gram.download()
    gram.free()
    var scatter: list[float] = []
    for i in 0..width {
        for j in 0..width { scatter.push((raw[i * width + j] + raw[j * width + i]) * 0.5) }
    }
    return (centre, scatter)
}

#: Each row of `data` less `centre`, times the transpose of `basis`, on the
#: device: the rows moved to the origin and written in the rows of `basis`.
#: Null when either is not a plain run of floats.
pub fn device_centred_product(data: Mat, centre: list[float], basis: Mat) -> Mat? {
    let x = device_view(data)
    if x is null { return null }
    let axes = device_view(basis)
    if axes is null {
        x.free()
        return null
    }
    var shift: list[float] = []
    for value in centre { shift.push(-value) }
    let offset = Tensor([centre.len()], values: shift)
    let centred = jt.add(x, offset)
    offset.free()
    x.free()
    let found = product(centred, axes, trans_b: true)
    centred.free()
    axes.free()
    return mat_of(found)
}
