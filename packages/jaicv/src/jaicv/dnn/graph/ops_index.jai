#: The indexing operators transformer exports build their masks and position
#: ids out of: `GatherElements`, `GatherND`, `ScatterND` and `Range`.
#:
#: They run on the host. In every model the zoo has met they act on a few
#: hundred integers -- position ids, an attention mask -- and most of them on
#: constants, which the plan folds away at compile time; a kernel apiece would
#: be four kernels to keep right for no measurable time. Each reads its inputs
#: down, works in Jaithon, and uploads one result.
import std.math as math
from jaitensor import Tensor

from .nodes import GraphNode, attr_int, attr_str, need
from .tensors import contiguous_strides, normalize_axis, volume

#: An index read out of a float tensor, counted from the end when negative,
#: and refused when it lands outside `width`.
fn index_into(value: float, width: int, node: GraphNode) -> int {
    var at = int(value)
    if at < 0 { at += width }
    if at < 0 or at >= width {
        throw ValueError(f"{node.label()} indexes {int(value)} on an axis {width} wide")
    }
    return at
}

#: `out[i][j][k] = data[i][indices[i][j][k]][k]` for `axis = 1`, and so on:
#: the output is the shape of the indices, and only the coordinate on `axis`
#: is replaced by what the indices say.
pub fn op_gather_elements(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let indices = need(ins, 1, node)
    let rank = data.rank()
    if indices.rank() != rank {
        throw ValueError(f"{node.label()} indexes a {data.shape} with {indices.shape}")
    }
    let axis = normalize_axis(attr_int(node, "axis", 0), rank, node.label())
    for other in 0..rank {
        if other != axis and indices.shape[other] > data.shape[other] {
            throw ValueError(f"{node.label()} indexes a {data.shape} with {indices.shape}")
        }
    }
    let values = data.download()
    let picks = indices.download()
    let from_strides = contiguous_strides(data.shape)
    let to_strides = contiguous_strides(indices.shape)
    var out: list[float] = []
    for flat in 0..picks.len() {
        var rest = flat
        var source = 0
        for other in 0..rank {
            var coordinate = rest // to_strides[other]
            rest = rest % to_strides[other]
            if other == axis { coordinate = index_into(picks[flat], data.shape[axis], node) }
            source += coordinate * from_strides[other]
        }
        out.push(values[source])
    }
    return [Tensor(indices.shape, out)]
}

#: `GatherND`: the last axis of the indices is a coordinate into the leading
#: axes of `data`, after the first `batch_dims` axes, which the two share.
pub fn op_gather_nd(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let indices = need(ins, 1, node)
    let batch = attr_int(node, "batch_dims", 0)
    let depth = indices.shape[indices.rank() - 1]
    if batch < 0 or batch + depth > data.rank() or batch >= indices.rank() {
        throw ValueError(f"{node.label()} indexes a {data.shape} with {indices.shape} past {batch} batch axes")
    }
    for axis in 0..batch {
        if indices.shape[axis] != data.shape[axis] {
            throw ValueError(f"{node.label()} has batch axes that differ: {data.shape} and {indices.shape}")
        }
    }
    var out_shape: list[int] = []
    for axis in 0..indices.rank() - 1 { out_shape.push(indices.shape[axis]) }
    var slab = 1
    for axis in batch + depth..data.rank() {
        out_shape.push(data.shape[axis])
        slab *= data.shape[axis]
    }
    let data_strides = contiguous_strides(data.shape)
    let values = data.download()
    let picks = indices.download()
    var per_batch = 1
    for axis in batch..indices.rank() - 1 { per_batch *= indices.shape[axis] }
    var out: list[float] = []
    for row in 0..picks.len() // depth {
        #: The batch this row belongs to fixes the leading coordinates.
        var base = 0
        var rest = row // per_batch
        var axis = batch - 1
        while axis >= 0 {
            base += rest % data.shape[axis] * data_strides[axis]
            rest = rest // data.shape[axis]
            axis -= 1
        }
        for step in 0..depth {
            let at = index_into(picks[row * depth + step], data.shape[batch + step], node)
            base += at * data_strides[batch + step]
        }
        for offset in 0..slab { out.push(values[base + offset]) }
    }
    return [Tensor(if out_shape.len() == 0 { [1] } else { out_shape }, out)]
}

#: `ScatterND`: a copy of `data` with the slices the indices name replaced by
#: `updates`, or combined with them by `reduction`.
pub fn op_scatter_nd(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let indices = need(ins, 1, node)
    let updates = need(ins, 2, node)
    let depth = indices.shape[indices.rank() - 1]
    if depth > data.rank() {
        throw ValueError(f"{node.label()} indexes a {data.shape} with {depth}-long coordinates")
    }
    let reduction = attr_str(node, "reduction", "none")
    if not reduction in ["none", "add", "mul", "max", "min"] {
        throw ValueError(f"{node.label()} asks for the {reduction} reduction, which is not implemented")
    }
    var slab = 1
    for axis in depth..data.rank() { slab *= data.shape[axis] }
    let rows = volume(indices.shape) // depth
    if updates.size != rows * slab {
        throw ValueError(f"{node.label()} has {updates.size} updates for {rows} slices of {slab}")
    }
    let strides = contiguous_strides(data.shape)
    var out = data.download()
    let picks = indices.download()
    let fresh = updates.download()
    for row in 0..rows {
        var base = 0
        for step in 0..depth {
            base += index_into(picks[row * depth + step], data.shape[step], node) * strides[step]
        }
        for offset in 0..slab {
            let value = fresh[row * slab + offset]
            let held = out[base + offset]
            out[base + offset] = match reduction {
                "add" => held + value,
                "mul" => held * value,
                "max" => max(held, value),
                "min" => min(held, value),
                _ => value,
            }
        }
    }
    return [Tensor(data.shape, out)]
}

#: `Range(start, limit, delta)`: `start, start + delta, ...` up to but not
#: including `limit`. An empty range has nowhere to go here and is refused.
pub fn op_range(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let start = need(ins, 0, node).download()[0]
    let limit = need(ins, 1, node).download()[0]
    let delta = need(ins, 2, node).download()[0]
    if delta == 0.0 { throw ValueError(f"{node.label()} steps by zero") }
    let count = int(math.ceil((limit - start) / delta))
    if count <= 0 {
        throw ValueError(f"{node.label()} is the empty range {start} to {limit}, and an empty tensor has no place here")
    }
    return [Tensor([count], [start + float(index) * delta for index in 0..count])]
}
