#: The arithmetic operators: activations, elementwise binaries and
#: comparisons, the variadic folds, matrix products, and the reductions and
#: softmaxes that run along a row.
import jaitensor as jt
from jaitensor import Tensor

from .kernels import HIGHEST_FLOAT, LOWEST_FLOAT
from .nodes import GraphNode, attr_float, attr_int, ints_from, maybe, need
from .tensors import binary_ex_of, broadcast_shape, copy_of, elementwise, expand_to
from .tensors import normalize_axis, row_reduce_of, shaped, softmax_rows, transpose_to
from .tensors import unary_ex_of, volume

pub fn op_relu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.relu(need(ins, 0, node))]
}

pub fn op_sigmoid(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.sigmoid(need(ins, 0, node))]
}

#: `x * sigmoid(x)`, which YOLOv8 and most modern detectors use everywhere.
#:
#: No exporter emits this: ONNX has no SiLU, so a file spells it as a Sigmoid
#: and a Mul. `fuse_activations` spots that pair and rewrites it to this, which
#: turns two passes over the tensor into one and drops the intermediate.
pub fn op_silu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.silu(need(ins, 0, node))]
}

pub fn op_tanh(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.tanh(need(ins, 0, node))]
}

pub fn op_sqrt(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.sqrt(need(ins, 0, node))]
}

pub fn op_exp(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.exp(need(ins, 0, node))]
}

pub fn op_log(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.log(need(ins, 0, node))]
}

pub fn op_abs(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.abs(need(ins, 0, node))]
}

pub fn op_neg(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.neg(need(ins, 0, node))]
}

pub fn op_elu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 0, first: attr_float(node, "alpha", 1.0))]
}

pub fn op_leaky_relu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 5, first: attr_float(node, "alpha", 0.01))]
}

pub fn op_hard_sigmoid(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let alpha = attr_float(node, "alpha", 0.2)
    let beta = attr_float(node, "beta", 0.5)
    return [unary_ex_of(need(ins, 0, node), 1, first: alpha, second: beta)]
}

pub fn op_softplus(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 2)]
}

pub fn op_erf(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 3)]
}

pub fn op_not(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 4)]
}

pub fn op_identity(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let input = need(ins, 0, node)
    return [copy_of(input, input.shape)]
}

#: Inference-time dropout is the identity, and the mask it can be asked for is
#: all ones because nothing was dropped.
pub fn op_dropout(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let input = need(ins, 0, node)
    var produced = [copy_of(input, input.shape)]
    if node.outputs.len() > 1 { produced.push(Tensor.ones(input.shape)) }
    return produced
}

pub fn op_prelu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let input = need(ins, 0, node)
    let slope = need(ins, 1, node)
    let stretched = if slope.shape == input.shape { slope } else { expand_to(slope, input.shape) }
    let output = binary_ex_of(input, stretched, 6)
    if stretched is not slope { stretched.free() }
    return [output]
}

pub fn op_clip(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let input = need(ins, 0, node)
    var low = attr_float(node, "min", LOWEST_FLOAT)
    var high = attr_float(node, "max", HIGHEST_FLOAT)
    let low_input = maybe(ins, 1)
    if low_input is not null {
        if low_input.size != 1 {
            throw ValueError(f"{node.label()} needs a single minimum, got {low_input.shape}")
        }
        low = low_input.download()[0]
    }
    let high_input = maybe(ins, 2)
    if high_input is not null {
        if high_input.size != 1 {
            throw ValueError(f"{node.label()} needs a single maximum, got {high_input.shape}")
        }
        high = high_input.download()[0]
    }
    return [jt.clamp(input, low, high)]
}

pub fn op_add(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), jt.add)]
}

pub fn op_sub(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), jt.subtract)]
}

pub fn op_mul(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), jt.multiply)]
}

pub fn op_div(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), jt.divide)]
}

pub fn op_pow(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 5)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

#: ONNX `Mod`: the remainder, with the sign following whichever operand the
#: `fmod` attribute says.
#:
#: Transformer exports reach for this in the arithmetic that works out head
#: offsets from a shape, so a graph runner without it stops at the first
#: attention block.
pub fn op_mod(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let code = if attr_int(node, "fmod", 0) != 0 { 8 } else { 7 }
    let combine = |a, b| binary_ex_of(a, b, code)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

pub fn op_equal(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 0)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

pub fn op_greater(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 1)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

pub fn op_less(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 2)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

pub fn op_and(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 3)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

pub fn op_or(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 4)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

#: Fold a variadic operator left to right, broadcasting at every step.
fn fold_inputs(
    node: GraphNode,
    ins: list[Tensor?],
    combine: fn(Tensor, Tensor) -> Tensor
) -> Tensor {
    if ins.len() == 0 { throw ValueError(f"{node.label()} needs at least one input") }
    var total = need(ins, 0, node)
    var mine = false
    for index in 1..ins.len() {
        let merged = elementwise(total, need(ins, index, node), node.label(), combine)
        if mine { total.free() }
        total = merged
        mine = true
    }
    if mine { return total }
    return copy_of(total, total.shape)
}

pub fn op_min(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [fold_inputs(node, ins, jt.minimum)]
}

pub fn op_max(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [fold_inputs(node, ins, jt.maximum)]
}

pub fn op_sum(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [fold_inputs(node, ins, jt.add)]
}

pub fn op_mean(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let total = fold_inputs(node, ins, jt.add)
    let averaged = jt.scale(total, 1.0 / float(ins.len()))
    total.free()
    return [averaged]
}

pub fn op_where(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let condition = need(ins, 0, node)
    let yes = need(ins, 1, node)
    let no = need(ins, 2, node)
    let shape = broadcast_shape(
        broadcast_shape(condition.shape, yes.shape, node.label()),
        no.shape,
        node.label()
    )
    let picked = if condition.shape == shape { condition } else { expand_to(condition, shape) }
    let left = if yes.shape == shape { yes } else { expand_to(yes, shape) }
    let right = if no.shape == shape { no } else { expand_to(no, shape) }
    let output = jt.where(picked, left, right)
    if picked is not condition { picked.free() }
    if left is not yes { left.free() }
    if right is not no { right.free() }
    return [output]
}

pub fn op_gemm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let a = need(ins, 0, node)
    let b = need(ins, 1, node)
    if a.rank() != 2 or b.rank() != 2 {
        throw ValueError(f"{node.label()} needs rank-two operands, got {a.shape} and {b.shape}")
    }
    let alpha = attr_float(node, "alpha", 1.0)
    let beta = attr_float(node, "beta", 1.0)
    let product = jt.matmul(
        a,
        b,
        trans_a: attr_int(node, "transA", 0) != 0,
        trans_b: attr_int(node, "transB", 0) != 0
    )
    var output = product
    if alpha != 1.0 {
        output = jt.scale(product, alpha)
        product.free()
    }
    let c = maybe(ins, 2)
    if c is null or beta == 0.0 { return [output] }
    let stretched = if c.shape == output.shape { c } else { expand_to(c, output.shape) }
    let weighted = if beta == 1.0 { stretched } else { jt.scale(stretched, beta) }
    let summed = jt.add(output, weighted)
    if weighted is not stretched { weighted.free() }
    if stretched is not c { stretched.free() }
    output.free()
    return [summed]
}

#: One `[rows, inner] x [inner, columns]` product per page of the broadcast
#: batch, written straight into a window of the output.
fn batched_matmul(
    left: Tensor,
    right: Tensor,
    batch: list[int],
    rows: int,
    inner: int,
    columns: int
) -> Tensor {
    var left_shape = [dimension for dimension in batch]
    left_shape.push(rows)
    left_shape.push(inner)
    var right_shape = [dimension for dimension in batch]
    right_shape.push(inner)
    right_shape.push(columns)
    var dims = [dimension for dimension in batch]
    dims.push(rows)
    dims.push(columns)
    let spread_left = if left.shape == left_shape { left } else { expand_to(left, left_shape) }
    let spread_right = if right.shape == right_shape {
        right
    } else {
        expand_to(right, right_shape)
    }
    let output = Tensor(dims)
    for index in 0..volume(batch) {
        let page_a = Tensor(
            [rows, inner],
            storage: spread_left.buffer().view(index * rows * inner, rows * inner)
        )
        let page_b = Tensor(
            [inner, columns],
            storage: spread_right.buffer().view(index * inner * columns, inner * columns)
        )
        let page_out = Tensor(
            [rows, columns],
            storage: output.buffer().view(index * rows * columns, rows * columns)
        )
        jt.matmul(page_a, page_b, out: page_out)
        page_a.free()
        page_b.free()
        page_out.free()
    }
    if spread_left is not left { spread_left.free() }
    if spread_right is not right { spread_right.free() }
    return output
}

pub fn op_matmul(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let a = need(ins, 0, node)
    let b = need(ins, 1, node)
    #: ONNX promotes a vector operand to a matrix for the product and drops
    #: the axis again afterwards, so both shapes are normalised here and the
    #: result is trimmed at the end.
    let left = if a.rank() == 1 { copy_of(a, [1, a.size]) } else { a }
    let right = if b.rank() == 1 { copy_of(b, [b.size, 1]) } else { b }
    let rows = left.shape[left.rank() - 2]
    let inner = left.shape[left.rank() - 1]
    let columns = right.shape[right.rank() - 1]
    if right.shape[right.rank() - 2] != inner {
        throw ValueError(f"{node.label()} cannot multiply {a.shape} by {b.shape}")
    }
    var batch: list[int] = []
    if left.rank() > 2 or right.rank() > 2 {
        batch = broadcast_shape(
            [left.shape[axis] for axis in 0..left.rank() - 2],
            [right.shape[axis] for axis in 0..right.rank() - 2],
            node.label()
        )
    }
    let product = if batch.len() == 0 {
        jt.matmul(left, right)
    } else {
        batched_matmul(left, right, batch, rows, inner, columns)
    }
    if left is not a { left.free() }
    if right is not b { right.free() }
    if a.rank() != 1 and b.rank() != 1 { return [product] }
    var trimmed = [dimension for dimension in batch]
    if a.rank() != 1 { trimmed.push(rows) }
    if b.rank() != 1 { trimmed.push(columns) }
    let squeezed = copy_of(product, shaped(trimmed))
    product.free()
    return [squeezed]
}

#: Reduce `axes` away, by moving them to the end and reducing the rows that
#: leaves. `op` is `row_reduce`'s: sum, mean, max, min, or an extreme's index.
fn reduce_over(x: Tensor, axes: list[int], keepdims: bool, op: int) -> Tensor {
    let rank = x.rank()
    var kept: list[int] = []
    for axis in 0..rank {
        if not axis in axes { kept.push(axis) }
    }
    var perm = [axis for axis in kept]
    for axis in axes { perm.push(axis) }
    var ordered = true
    for (slot, axis) in perm.enumerate() {
        if slot != axis { ordered = false }
    }
    let moved = if ordered { x } else { transpose_to(x, perm) }
    var inner = 1
    for axis in axes { inner *= x.shape[axis] }
    var dims: list[int] = []
    if keepdims {
        for axis in 0..rank { dims.push(if axis in axes { 1 } else { x.shape[axis] }) }
    } else {
        for axis in kept { dims.push(x.shape[axis]) }
    }
    let output = row_reduce_of(moved, x.size // inner, inner, op, shape: shaped(dims))
    if moved is not x { moved.free() }
    return output
}

fn sorted_axes(node: GraphNode, asked: list[int], rank: int) -> list[int] {
    var resolved: list[int] = []
    for axis in asked {
        let normalized = normalize_axis(axis, rank, node.label())
        if not normalized in resolved { resolved.push(normalized) }
    }
    resolved.sort()
    return resolved
}

fn op_reduce(node: GraphNode, ins: list[Tensor?], op: int) -> list[Tensor] {
    let x = need(ins, 0, node)
    let asked = ints_from(node, ins, 1, "axes") ?? []
    if asked.len() == 0 and attr_int(node, "noop_with_empty_axes", 0) != 0 {
        return [copy_of(x, x.shape)]
    }
    let wanted = if asked.len() == 0 { [axis for axis in 0..x.rank()] } else { asked }
    let axes = sorted_axes(node, wanted, x.rank())
    return [reduce_over(x, axes, attr_int(node, "keepdims", 1) != 0, op)]
}

pub fn op_reduce_sum(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_reduce(node, ins, 0)
}

pub fn op_reduce_mean(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_reduce(node, ins, 1)
}

pub fn op_reduce_max(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_reduce(node, ins, 2)
}

pub fn op_reduce_min(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_reduce(node, ins, 3)
}

fn op_arg_extreme(node: GraphNode, ins: list[Tensor?], want_max: bool) -> list[Tensor] {
    let x = need(ins, 0, node)
    let axis = normalize_axis(attr_int(node, "axis", 0), x.rank(), node.label())
    var op = if want_max { 4 } else { 5 }
    if attr_int(node, "select_last_index", 0) != 0 { op += 2 }
    return [reduce_over(x, [axis], attr_int(node, "keepdims", 1) != 0, op)]
}

pub fn op_argmax(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_arg_extreme(node, ins, true)
}

pub fn op_argmin(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_arg_extreme(node, ins, false)
}

fn softmax_like(node: GraphNode, ins: list[Tensor?], want_log: bool) -> list[Tensor] {
    let x = need(ins, 0, node)
    let rank = x.rank()
    let axis = normalize_axis(attr_int(node, "axis", -1), rank, node.label())
    let columns = x.shape[axis]
    if axis == rank - 1 { return [softmax_rows(x, x.size // columns, columns, want_log)] }
    var perm: list[int] = []
    for other in 0..rank {
        if other != axis { perm.push(other) }
    }
    perm.push(axis)
    let moved = transpose_to(x, perm)
    let normalized = softmax_rows(moved, x.size // columns, columns, want_log)
    moved.free()
    var back = [0 for _axis in 0..rank]
    for (slot, other) in perm.enumerate() { back[other] = slot }
    let output = transpose_to(normalized, back)
    normalized.free()
    return [output]
}

pub fn op_softmax(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return softmax_like(node, ins, false)
}

pub fn op_log_softmax(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return softmax_like(node, ins, true)
}
