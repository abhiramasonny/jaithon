#: Resolving the bounds a slice asks for and the padding a window asks for
#: against the shapes the survey saw.
from .build import Build, as_ints
from ..graph import GraphNode, attr_ints, attr_str, normalize_axis

#: A slice bound out of the float it was folded into. An open end is
#: written as the largest int64, which as a float is one past it and does not
#: convert back; anything that far out means the end of the axis anyway.
fn bound_of(value: float) -> int {
    if value >= 4.0e18 { return 1 << 62 }
    if value <= -4.0e18 { return -(1 << 62) }
    return int(value)
}

#: A `Slice` node's per-axis bounds, resolved against the input's real shape.
#:
#: ONNX writes an open end as the largest integer there is and allows negative
#: indices; the builder wants neither, so both are settled here. Steps may be
#: greater than one, and whatever comes out is checked against the shape the
#: survey saw before any of it is used -- which is what makes compiling the
#: awkward cases safe rather than hopeful.
pub fn slice_bounds(
    state: Build,
    node: GraphNode,
    shape: list[int],
    want: list[int]
) -> list[list[int]]? {
    let rank = shape.len()
    var starts = [0 for _index in 0..rank]
    var ends = [shape[index] for index in 0..rank]
    var steps = [1 for _index in 0..rank]

    let given_starts = state.constants.get(node.inputs[1], null)
    let given_ends = if node.inputs.len() > 2 {
        state.constants.get(node.inputs[2], null)
    } else {
        null
    }
    if given_starts is null or given_ends is null { return null }
    var axes: list[int] = []
    if node.inputs.len() > 3 and node.inputs[3].len() > 0 {
        let given = state.constants.get(node.inputs[3], null)
        if given is null { return null }
        axes = as_ints(given)
    } else {
        for index in 0..given_starts.len() { axes.push(index) }
    }
    var given_steps: list[int] = []
    for _index in 0..axes.len() { given_steps.push(1) }
    if node.inputs.len() > 4 and node.inputs[4].len() > 0 {
        let given = state.constants.get(node.inputs[4], null)
        if given is null { return null }
        given_steps = as_ints(given)
        if given_steps.len() != axes.len() { return null }
    }

    for (slot, raw_axis) in axes.enumerate() {
        let axis = normalize_axis(raw_axis, rank, node.label())
        let width = shape[axis]
        let step = given_steps[slot]
        #: Forward only. A negative stride means something different here than
        #: it does in ONNX -- the operation reads the end as an index from the
        #: back rather than as one past the beginning, and a reversed slice
        #: came out empty -- so those go to the interpreter.
        if step <= 0 { return null }
        var first = bound_of(given_starts[slot])
        var last = bound_of(given_ends[slot])
        if first < 0 { first += width }
        if last < 0 { last += width }
        #: Forward and backward clamp to different ends: walking down, the
        #: last index reachable is zero and "past the start" is -1.
        first = min(max(first, 0), width)
        last = min(max(last, 0), width)
        starts[axis] = first
        ends[axis] = last
        steps[axis] = step
    }
    #: The survey already knows what came out; if these bounds disagree with
    #: it, something has been read wrongly and the interpreter should have it.
    #: That check is what makes it safe to compile the awkward cases at all.
    for index in 0..rank {
        let step = steps[index]
        let span = (ends[index] - starts[index] + step - 1) // step
        if max(span, 0) != want[index] { return null }
        #: An empty slice has no tensor in the compiled graph to land in --
        #: MPSGraph hands back a width of one -- so it is the interpreter's.
        if want[index] == 0 { return null }
    }
    return [starts, ends, steps]
}

#: A convolution or pooling node's four padding values, or nothing when the
#: node asks for a padding this compiler does not express.
pub fn resolved_pads(
    node: GraphNode,
    spatial: list[int],
    window: list[int],
    strides: list[int],
    dilations: list[int]
) -> list[int]? {
    #: As ONNX writes them, every axis's start and then every axis's end;
    #: two axes for an image and one for a sequence.
    let axes = spatial.len()
    let none = [0 for _slot in 0..2 * axes]
    let mode = attr_str(node, "auto_pad", "NOTSET")
    if mode == "NOTSET" {
        let pads = attr_ints(node, "pads", none)
        if pads.len() != 2 * axes { return null }
        return pads
    }
    if mode == "VALID" { return none }
    if mode != "SAME_UPPER" and mode != "SAME_LOWER" { return null }
    var pads = [0 for _slot in 0..2 * axes]
    for axis in 0..axes {
        let reach = (window[axis] - 1) * dilations[axis] + 1
        let out = (spatial[axis] + strides[axis] - 1) // strides[axis]
        let total = max((out - 1) * strides[axis] + reach - spatial[axis], 0)
        let head = if mode == "SAME_UPPER" { total // 2 } else { total - total // 2 }
        pads[axis] = head
        pads[axis + axes] = total - head
    }
    return pads
}
