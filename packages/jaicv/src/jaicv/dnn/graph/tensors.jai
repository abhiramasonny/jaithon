#: The shape arithmetic and the tensor primitives the operators are built
#: out of: strides and broadcasting, the one strided read that every layout
#: change goes through, and the pads, joins and row reductions that have a
#: kernel of their own.
from jaitensor import Tensor

from .kernels import MAX_RANK, f32_bits, kernel_for, u32

#: Number of elements a shape holds.
pub fn volume(shape: list[int]) -> int {
    var total = 1
    for dimension in shape { total *= dimension }
    return total
}

#: Row-major strides for a shape, in elements.
pub fn contiguous_strides(shape: list[int]) -> list[int] {
    var strides = [0 for _dimension in shape]
    var running = 1
    var axis = shape.len() - 1
    while axis >= 0 {
        strides[axis] = running
        running *= shape[axis]
        axis -= 1
    }
    return strides
}

#: ONNX has rank-zero tensors and jaitensor does not, so a scalar is carried
#: as `[1]`. It broadcasts the same way, which is the only thing that ever
#: looks at the difference.
pub fn shaped(dims: list[int]) -> list[int] {
    if dims.len() == 0 { return [1] }
    return dims
}

fn check_rank(rank: int, what: str) -> void {
    if rank > MAX_RANK {
        throw ValueError(f"{what} needs rank {rank}, and this executor indexes at most {MAX_RANK}")
    }
}

#: A fresh contiguous copy of `input`, reinterpreted as `shape`.
pub fn copy_of(input: Tensor, shape: list[int]) -> Tensor {
    let output = Tensor(shape)
    if output.size != input.size {
        throw ValueError(f"cannot view {input.shape} ({input.size} values) as {shape}")
    }
    kernel_for("copy_flat").dispatch_async(
        [input.buffer(), output.buffer()],
        input.size,
        [input.size]
    )
    return output
}

#: Read `input` through `base` and `strides` and write the result out dense.
#:
#: This one kernel is every layout change the executor makes: a transpose is a
#: permuted stride list, a slice is an offset base and a scaled stride, and a
#: broadcast is a stride of zero.
pub fn gather_strided(input: Tensor, shape: list[int], base: int, strides: list[int]) -> Tensor {
    let rank = shape.len()
    check_rank(rank, "a strided read")
    let output = Tensor(shape)
    var scalars = [output.size, rank, u32(base)]
    for axis in 0..MAX_RANK { scalars.push(if axis < rank { shape[axis] } else { 1 }) }
    for axis in 0..MAX_RANK { scalars.push(u32(if axis < rank { strides[axis] } else { 0 })) }
    kernel_for("strided_gather").dispatch_async(
        [input.buffer(), output.buffer()],
        output.size,
        scalars
    )
    return output
}

#: `input` with its axes reordered so axis `perm[k]` becomes axis `k`.
pub fn transpose_to(input: Tensor, perm: list[int]) -> Tensor {
    let source = contiguous_strides(input.shape)
    var shape: list[int] = []
    var strides: list[int] = []
    for axis in perm {
        shape.push(input.shape[axis])
        strides.push(source[axis])
    }
    return gather_strided(input, shape, 0, strides)
}

#: A contiguous window of `input`, `length` wide, starting at `start` on `axis`.
pub fn slice_axis(input: Tensor, axis: int, start: int, length: int) -> Tensor {
    let strides = contiguous_strides(input.shape)
    var shape = [dimension for dimension in input.shape]
    shape[axis] = length
    return gather_strided(input, shape, start * strides[axis], strides)
}

#: The shape numpy would broadcast `a` and `b` to.
pub fn broadcast_shape(a: list[int], b: list[int], what: str) -> list[int] {
    let rank = max(a.len(), b.len())
    let left_pad = rank - a.len()
    let right_pad = rank - b.len()
    var shape: list[int] = []
    for axis in 0..rank {
        let left = if axis < left_pad { 1 } else { a[axis - left_pad] }
        let right = if axis < right_pad { 1 } else { b[axis - right_pad] }
        if left == right or right == 1 {
            shape.push(left)
        } elif left == 1 {
            shape.push(right)
        } else {
            throw ValueError(f"{what} cannot broadcast {a} against {b}")
        }
    }
    return shape
}

#: `input` stretched to `shape` by numpy's rules, with a stride of zero on
#: every axis being repeated so nothing is actually copied more than once.
pub fn expand_to(input: Tensor, shape: list[int]) -> Tensor {
    let offset = shape.len() - input.rank()
    if offset < 0 {
        throw ValueError(f"cannot broadcast {input.shape} up to the shorter {shape}")
    }
    let source = contiguous_strides(input.shape)
    var strides = [0 for _dimension in shape]
    for axis in 0..input.rank() {
        let dimension = input.shape[axis]
        let target = shape[offset + axis]
        if dimension == target {
            strides[offset + axis] = source[axis]
        } elif dimension != 1 {
            throw ValueError(f"cannot broadcast {input.shape} up to {shape}")
        }
    }
    return gather_strided(input, shape, 0, strides)
}

#: Apply `combine` to `a` and `b` after stretching both to their common shape.
#:
#: Frees only what it stretched, so an operand that needed no stretching is
#: handed straight through and is still the caller's.
pub fn elementwise(
    a: Tensor,
    b: Tensor,
    what: str,
    combine: fn(Tensor, Tensor) -> Tensor
) -> Tensor {
    let shape = broadcast_shape(a.shape, b.shape, what)
    let left = if a.shape == shape { a } else { expand_to(a, shape) }
    let right = if b.shape == shape { b } else { expand_to(b, shape) }
    let output = combine(left, right)
    if left is not a { left.free() }
    if right is not b { right.free() }
    return output
}

pub fn binary_ex_of(a: Tensor, b: Tensor, op: int) -> Tensor {
    let output = Tensor(a.shape)
    kernel_for("binary_ex").dispatch_async(
        [a.buffer(), b.buffer(), output.buffer()],
        a.size,
        [a.size, op]
    )
    return output
}

pub fn unary_ex_of(input: Tensor, op: int, first: float = 0.0, second: float = 0.0) -> Tensor {
    let output = Tensor(input.shape)
    kernel_for("unary_ex").dispatch_async(
        [input.buffer(), output.buffer()],
        input.size,
        [input.size, op, f32_bits(first), f32_bits(second)]
    )
    return output
}

#: Reduce the last axis of a `[rows, columns]` view. `op` picks sum, mean, max,
#: min, or the index of the first or last extreme.
pub fn row_reduce_of(
    input: Tensor,
    rows: int,
    columns: int,
    op: int,
    shape: list[int]? = null
) -> Tensor {
    let output = Tensor(shape ?? [rows])
    kernel_for("row_reduce").dispatch_async(
        [input.buffer(), output.buffer()],
        rows,
        [rows, columns, op]
    )
    return output
}

pub fn softmax_rows(input: Tensor, rows: int, columns: int, want_log: bool) -> Tensor {
    let output = Tensor(input.shape)
    let logging = if want_log { 1 } else { 0 }
    kernel_for("row_softmax_ex").dispatch_async(
        [input.buffer(), output.buffer()],
        rows,
        [rows, columns, logging]
    )
    return output
}

#: Grow (or, with negative counts, crop) every axis. `mode` is 0 constant,
#: 1 reflect, 2 edge.
pub fn pad_nd_of(
    input: Tensor,
    before: list[int],
    after: list[int],
    mode: int,
    value: float
) -> Tensor {
    let rank = input.rank()
    check_rank(rank, "a pad")
    var shape: list[int] = []
    for axis in 0..rank {
        let width = input.shape[axis] + before[axis] + after[axis]
        if width <= 0 {
            throw ValueError(f"padding {before}/{after} leaves {input.shape} empty on axis {axis}")
        }
        shape.push(width)
    }
    let output = Tensor(shape)
    var scalars = [output.size, rank, mode, f32_bits(value)]
    for axis in 0..MAX_RANK { scalars.push(if axis < rank { shape[axis] } else { 1 }) }
    for axis in 0..MAX_RANK { scalars.push(if axis < rank { input.shape[axis] } else { 1 }) }
    for axis in 0..MAX_RANK { scalars.push(u32(if axis < rank { before[axis] } else { 0 })) }
    kernel_for("pad_nd").dispatch_async([input.buffer(), output.buffer()], output.size, scalars)
    return output
}

#: Join tensors along `axis`. Every part must agree on every other axis.
pub fn concat_along(parts: list[Tensor], axis: int, what: str) -> Tensor {
    if parts.len() == 0 { throw ValueError(f"{what} needs at least one input") }
    let rank = parts[0].rank()
    var width = 0
    for part in parts {
        if part.rank() != rank {
            throw ValueError(f"{what} joins ranks {parts[0].shape} and {part.shape}")
        }
        for other in 0..rank {
            if other == axis or part.size == 0 { continue }
            if part.shape[other] != parts[0].shape[other] {
                throw ValueError(f"{what} cannot join {parts[0].shape} and {part.shape}")
            }
        }
        width += part.shape[axis]
    }
    var shape = [dimension for dimension in parts[0].shape]
    shape[axis] = width
    let output = Tensor(shape)
    if output.size == 0 { return output }
    var inner = 1
    for other in axis + 1..rank { inner *= shape[other] }
    var offset = 0
    let kernel = kernel_for("concat_into")
    for part in parts {
        #: An empty part contributes nothing and moves the offset nowhere.
        if part.size == 0 { continue }
        kernel.dispatch_async(
            [part.buffer(), output.buffer()],
            part.size,
            [part.size, part.shape[axis], inner, width, offset]
        )
        offset += part.shape[axis]
    }
    return output
}

#: The largest index this executor will read out of a tensor.
#:
#: An exported model writes `INT64_MAX` for "to the end of this axis", and
#: that number does not survive a float32 tensor -- it comes back a shade
#: above what an `int` holds and overflows the conversion. Every use these
#: integers have is a bound, a shape or an index, and saturating is the right
#: answer for all three.
const SATURATED_INDEX = 1 << 53

#: The integers a tensor of shape or index values holds.
pub fn host_ints(input: Tensor) -> list[int] {
    let ceiling = float(SATURATED_INDEX)
    var values: list[int] = []
    for value in input.download() {
        if value >= ceiling {
            values.push(SATURATED_INDEX)
        } elif value <= -ceiling {
            values.push(-SATURATED_INDEX)
        } else {
            values.push(int(value))
        }
    }
    return values
}

pub fn normalize_axis(axis: int, rank: int, what: str) -> int {
    let resolved = if axis < 0 { rank + axis } else { axis }
    if resolved < 0 or resolved >= rank {
        throw ValueError(f"{what} names axis {axis}, which is outside rank {rank}")
    }
    return resolved
}
