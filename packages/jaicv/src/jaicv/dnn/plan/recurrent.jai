#: The recurrent operators, onto the platform's own LSTM and GRU.
from .build import Build
from ..graph import GraphNode, attr_int, attr_str

#: One direction's worth of an ONNX weight block, with the gates put into the
#: order the platform's own LSTM wants.
#:
#: ONNX orders them i, o, f, z along the first axis and the operation orders
#: them i, f, z, o, so the blocks move `[0, 2, 3, 1]`. Doing it here, on the
#: initializer's numbers, means it happens once at compile time instead of
#: being a transpose in the graph.
fn regated(values: list[float], direction: int, hidden: int, width: int) -> list[float] {
    let block = hidden * width
    let base = direction * 4 * block
    var out = __prim__.list_filled(4 * block, 0.0)
    var at = 0
    for source in [0, 2, 3, 1] {
        for index in 0..block { out[at + index] = values[base + source * block + index] }
        at += block
    }
    return out
}

#: ONNX `LSTM` as the platform's single operation.
#:
#: Only what maps exactly: forward or reverse, no peepholes, no per-sequence
#: lengths, the default activations. Anything else is left to the interpreter,
#: which implements the whole operator.
pub fn emit_lstm(state: Build, node: GraphNode) -> bool {
    let b = state.builder
    if node.outputs.len() == 0 { return false }
    #: One direction at a time. A reversed pass comes back in the order it ran
    #: rather than indexed by timestep, which is put right below.
    #:
    #: Bidirectional is left to the interpreter. The operation has a flag for
    #: it and the shapes its header describes line up with what ONNX carries --
    #: `[2, 4H, H]` recurrent weights, `[8H, I]` input weights, `[8H]` bias,
    #: `[N, 2H]` initial state, `[T, N, 2H]` out -- but feeding it those gave an
    #: answer half a unit off the reference, and neither swapping the two
    #: halves of the channel axis nor turning either of them around in time
    #: brought it closer.
    #:
    #: Tried twice. The second attempt, laying the directions out with a
    #: transpose and slicing each pass's last state from that, got MPSGraph to
    #: fail its own verifier and abort the process rather than return a wrong
    #: answer -- which is what makes this worth refusing rather than
    #: approximating: a wrong guess here takes the program down. The
    #: interpreter runs the operator correctly, so what is lost is speed on a
    #: model that uses it, not the model.
    let heading = attr_str(node, "direction", "forward")
    if heading != "forward" and heading != "reverse" { return false }
    let backwards = heading == "reverse"
    if node.attrs.get("activations", null) is not null { return false }
    #: The operation has no gate clamp, and ignoring one that a file asks for
    #: would quietly change the answer.
    if node.attrs.get("clip", null) is not null { return false }
    if attr_int(node, "layout", 0) != 0 or attr_int(node, "input_forget", 0) != 0 { return false }
    if node.inputs.len() > 4 and node.inputs[4].len() > 0 { return false }
    if node.inputs.len() > 7 and node.inputs[7].len() > 0 { return false }

    let x = state.shape_of(node.inputs[0])
    if x is null or x.len() != 3 { return false }
    let steps = x[0]
    let batch = x[1]
    let input_size = x[2]
    let weights = state.numbers_of(node.inputs[1])
    let recurrent = state.numbers_of(node.inputs[2])
    #: The weights have to be numbers here, since the gates are reordered.
    if weights is null or recurrent is null { return false }
    let w_shape = state.shape_of(node.inputs[1])
    if w_shape is null or w_shape.len() != 3 { return false }
    let hidden = w_shape[1] // 4
    if hidden <= 0 or w_shape[1] != 4 * hidden { return false }

    let w = __prim__.graph_constant(
        b,
        regated(weights, 0, hidden, input_size),
        [4 * hidden, input_size]
    )
    let r = __prim__.graph_constant(b, regated(recurrent, 0, hidden, hidden), [4 * hidden, hidden])

    #: ONNX carries an input bias and a recurrent one; the operation takes a
    #: single bias added to the same gates, so their sum is the same thing.
    var bias = -1
    if node.inputs.len() > 3 and node.inputs[3].len() > 0 {
        let given = state.numbers_of(node.inputs[3])
        if given is null { return false }
        var summed = __prim__.list_filled(4 * hidden, 0.0)
        for index in 0..4 * hidden { summed[index] = given[index] + given[4 * hidden + index] }
        bias = __prim__.graph_constant(b, regated(summed, 0, hidden, 1), [4 * hidden])
    }

    #: The initial state and cell arrive as `[1, N, H]` and are wanted `[N, H]`.
    var start_state = -1
    var start_cell = -1
    if node.inputs.len() > 5 and node.inputs[5].len() > 0 {
        start_state = __prim__.graph_reshape(b, state.id_of(node.inputs[5]), [batch, hidden])
    }
    if node.inputs.len() > 6 and node.inputs[6].len() > 0 {
        start_cell = __prim__.graph_reshape(b, state.id_of(node.inputs[6]), [batch, hidden])
    }

    let made = __prim__.graph_lstm(b, state.id_of(node.inputs[0]), r, w, bias, start_state, start_cell, if backwards {
        1
    } else {
        0
    })
    if made.len() != 2 { return false }
    #: Turned back into timestep order for a reversed pass, which then puts the
    #: last step computed at the front rather than the end.
    let all_state = if backwards { flipped_time(state, made[0], steps) } else { made[0] }
    let all_cell = if backwards { flipped_time(state, made[1], steps) } else { made[1] }
    if all_state < 0 or all_cell < 0 { return false }

    #: `Y` is every timestep with a direction axis put back in; `Y_h` and `Y_c`
    #: are the last step the pass produced.
    if node.outputs[0].len() > 0 {
        state.ids[node.outputs[0]] = __prim__.graph_reshape(b, all_state, [steps, 1, batch, hidden])
    }
    let last = if backwards { 0 } else { steps - 1 }
    if node.outputs.len() > 1 and node.outputs[1].len() > 0 {
        let cut = __prim__.graph_slice(
            b,
            all_state,
            [last, 0, 0],
            [last + 1, batch, hidden],
            [1, 1, 1]
        )
        state.ids[node.outputs[1]] = __prim__.graph_reshape(b, cut, [1, batch, hidden])
    }
    if node.outputs.len() > 2 and node.outputs[2].len() > 0 {
        let cut = __prim__.graph_slice(
            b,
            all_cell,
            [last, 0, 0],
            [last + 1, batch, hidden],
            [1, 1, 1]
        )
        state.ids[node.outputs[2]] = __prim__.graph_reshape(b, cut, [1, batch, hidden])
    }
    return true
}

#: A `[T, ...]` tensor with its time axis turned around.
#:
#: There is no negative-step slice in the builder, so the reversal is a gather
#: over a constant order, which is what one would compile to anyway.
fn flipped_time(state: Build, id: int, steps: int) -> int {
    let order = __prim__.graph_constant(
        state.builder,
        [float(steps - 1 - index) for index in 0..steps],
        [steps]
    )
    if order < 0 { return -1 }
    return __prim__.graph_gather(state.builder, id, order, 0)
}

#: ONNX `GRU` as the platform's single operation.
#:
#: Both reset variants map. `linear_before_reset` -- what PyTorch exports, and
#: so what most GRUs in the wild are -- puts the candidate gate's recurrent
#: bias inside the reset while its input bias stays outside; the operation
#: takes that one apart as a second bias and the other two gates keep both of
#: theirs summed together.
pub fn emit_gru(state: Build, node: GraphNode) -> bool {
    let b = state.builder
    if node.outputs.len() == 0 { return false }
    let heading = attr_str(node, "direction", "forward")
    if heading != "forward" and heading != "reverse" { return false }
    if node.attrs.get("activations", null) is not null { return false }
    if node.attrs.get("clip", null) is not null { return false }
    if attr_int(node, "layout", 0) != 0 { return false }
    if node.inputs.len() > 4 and node.inputs[4].len() > 0 { return false }
    let backwards = heading == "reverse"

    let x = state.shape_of(node.inputs[0])
    if x is null or x.len() != 3 { return false }
    let steps = x[0]
    let batch = x[1]
    let w_shape = state.shape_of(node.inputs[1])
    let r_shape = state.shape_of(node.inputs[2])
    if w_shape is null or r_shape is null { return false }
    if w_shape.len() != 3 or r_shape.len() != 3 { return false }
    let hidden = r_shape[2]
    if w_shape[1] != 3 * hidden { return false }

    #: One direction, so the leading axis is dropped rather than sliced.
    let w = __prim__.graph_reshape(b, state.id_of(node.inputs[1]), [3 * hidden, w_shape[2]])
    let r = __prim__.graph_reshape(b, state.id_of(node.inputs[2]), [3 * hidden, hidden])

    let linear = attr_int(node, "linear_before_reset", 0) != 0
    var bias = -1
    var reset_bias = -1
    if node.inputs.len() > 3 and node.inputs[3].len() > 0 {
        let given = state.numbers_of(node.inputs[3])
        if given is null { return false }
        #: The update and reset gates carry both biases outside the reset, so
        #: their sum is what the operation wants. The candidate gate follows
        #: them only when the reset comes first; with `linear_before_reset` its
        #: recurrent half sits inside the reset and travels separately.
        var summed = __prim__.list_filled(3 * hidden, 0.0)
        for index in 0..3 * hidden { summed[index] = given[index] + given[3 * hidden + index] }
        if linear {
            var inside = __prim__.list_filled(hidden, 0.0)
            for index in 0..hidden {
                summed[2 * hidden + index] = given[2 * hidden + index]
                inside[index] = given[5 * hidden + index]
            }
            reset_bias = __prim__.graph_constant(b, inside, [hidden])
        }
        bias = __prim__.graph_constant(b, summed, [3 * hidden])
    }
    var start = -1
    if node.inputs.len() > 5 and node.inputs[5].len() > 0 {
        start = __prim__.graph_reshape(b, state.id_of(node.inputs[5]), [batch, hidden])
    }

    let all_state = __prim__.graph_gru(b, state.id_of(node.inputs[0]), r, w, bias, start, if backwards {
        1
    } else {
        0
    }, if linear {
        1
    } else {
        0
    }, reset_bias)
    if all_state < 0 { return false }
    #: A reversed pass comes back in the order it ran, last input first. ONNX
    #: indexes `Y` by the timestep the input came from, so the sequence is
    #: turned around again -- and the final state, which is the last one
    #: computed, ends up at the front of what that leaves.
    let ordered = if backwards { flipped_time(state, all_state, steps) } else { all_state }
    if ordered < 0 { return false }
    if node.outputs[0].len() > 0 {
        state.ids[node.outputs[0]] = __prim__.graph_reshape(b, ordered, [steps, 1, batch, hidden])
    }
    if node.outputs.len() > 1 and node.outputs[1].len() > 0 {
        let last = if backwards { 0 } else { steps - 1 }
        let cut = __prim__.graph_slice(
            b,
            ordered,
            [last, 0, 0],
            [last + 1, batch, hidden],
            [1, 1, 1]
        )
        state.ids[node.outputs[1]] = __prim__.graph_reshape(b, cut, [1, batch, hidden])
    }
    return true
}
