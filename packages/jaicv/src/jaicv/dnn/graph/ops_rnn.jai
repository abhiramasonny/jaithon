#: The recurrent operators, `LSTM` and `GRU`, and the per-direction walk each
#: of them is built out of.
import jaitensor as jt
from jaitensor import Tensor

from .kernels import f32_bits, kernel_for
from .nodes import GraphNode, attr_float, attr_int, attr_str, maybe, need
from .tensors import concat_along, copy_of, slice_axis

#: One direction of an LSTM, from `[seq, batch, input]` to `[seq, batch, H]`.
#:
#: `x_gates` already holds every timestep's `Xt W' + b`, because that half does
#: not depend on the recurrence and so is one matrix product for the whole
#: sequence rather than one per step. What is left per step is the recurrent
#: product and the gates.
fn lstm_direction(
    x_gates: Tensor,
    recurrent: Tensor,
    initial_h: Tensor,
    initial_c: Tensor,
    steps: int,
    batch: int,
    hidden: int,
    clip: float,
    reverse: bool
) -> tuple[Tensor, Tensor, Tensor] {
    #: `steps` owns every hidden state it produces, because they are also the
    #: output sequence. The cell state is not kept, so each new one replaces
    #: and frees the one before -- except the caller's, which the caller frees.
    var state_h = initial_h
    var state_c = initial_c
    var own_cell = false
    var produced: list[Tensor] = []
    for index in 0..steps {
        let step = if reverse { steps - 1 - index } else { index }
        let recurred = jt.matmul(state_h, recurrent, trans_b: true)
        let next_c = jt.Tensor([batch, hidden])
        let next_h = jt.Tensor([batch, hidden])
        kernel_for("lstm_gates").dispatch_async(
            [
                x_gates.buffer(),
                recurred.buffer(),
                state_c.buffer(),
                next_c.buffer(),
                next_h.buffer(),
            ],
            batch * hidden,
            [batch * hidden, hidden, f32_bits(clip), step * batch * 4 * hidden]
        )
        recurred.free()
        if own_cell { state_c.free() }
        state_c = next_c
        own_cell = true
        state_h = next_h
        produced.push(next_h)
    }

    #: A reversed pass produces its steps back to front, and the output is
    #: indexed by time whichever way it was walked.
    var ordered = produced
    if reverse {
        ordered = []
        for index in 0..produced.len() { ordered.push(produced[produced.len() - 1 - index]) }
    }
    var rows: list[Tensor] = []
    for part in ordered { rows.push(copy_of(part, [1, batch, hidden])) }
    let sequence = if rows.len() == 1 {
        rows[0]
    } else {
        concat_along(rows, 0, "an LSTM's output sequence")
    }
    if rows.len() > 1 {
        for row in rows { row.free() }
    }
    let final_h = copy_of(state_h, [batch, hidden])
    let final_c = copy_of(state_c, [batch, hidden])
    for part in produced { part.free() }
    if own_cell { state_c.free() }
    return (sequence, final_h, final_c)
}

#: ONNX `LSTM`.
#:
#: Weights arrive as `[directions, 4H, input]` and `[directions, 4H, H]` with
#: the gates ordered i, o, f, c, and the bias as `[directions, 8H]` holding the
#: input biases and then the recurrent ones. Outputs are the whole sequence
#: `[seq, directions, batch, H]` and the final hidden and cell states.
pub fn op_lstm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let w = need(ins, 1, node)
    let r = need(ins, 2, node)
    let bias = maybe(ins, 3)
    if maybe(ins, 4) is not null {
        throw ValueError(
            f"{node.label()} gives per-sequence lengths, which this executor does not read; " +
                "every sequence in the batch has to be the same length"
        )
    }
    if maybe(ins, 7) is not null {
        throw ValueError(f"{node.label()} has peephole weights, which this executor does not apply")
    }
    if node.attrs.get("activations", null) is not null {
        throw ValueError(
            f"{node.label()} names its own activations; this executor uses the sigmoid and " +
                "tanh the operator defaults to"
        )
    }
    if attr_int(node, "layout", 0) != 0 {
        throw ValueError(
            f"{node.label()} uses the batch-first layout, which this executor does not read"
        )
    }
    if attr_int(node, "input_forget", 0) != 0 {
        throw ValueError(
            f"{node.label()} couples its input and forget gates, which this executor does not"
        )
    }
    if x.rank() != 3 { throw ValueError(f"{node.label()} needs a 3-D input, got {x.shape}") }
    if w.rank() != 3 or r.rank() != 3 {
        throw ValueError(f"{node.label()} needs 3-D weights, got {w.shape} and {r.shape}")
    }
    let steps = x.shape[0]
    let batch = x.shape[1]
    let input_size = x.shape[2]
    let directions = w.shape[0]
    let hidden = attr_int(node, "hidden_size", r.shape[2])
    let facing = attr_str(node, "direction", "forward")
    let wanted = if facing == "bidirectional" { 2 } else { 1 }
    if directions != wanted or r.shape[0] != directions {
        throw ValueError(
            f"{node.label()} is {facing} but carries {directions} directions of weights"
        )
    }
    if w.shape[1] != 4 * hidden or r.shape[1] != 4 * hidden or r.shape[2] != hidden {
        throw ValueError(
            f"{node.label()} has weights {w.shape} and {r.shape} for {hidden} hidden units"
        )
    }
    if w.shape[2] != input_size {
        throw ValueError(f"{node.label()} has weights {w.shape} for {input_size} inputs")
    }
    let clip = attr_float(node, "clip", 0.0)

    let flat_x = copy_of(x, [steps * batch, input_size])
    var sequences: list[Tensor] = []
    var last_h: list[Tensor] = []
    var last_c: list[Tensor] = []
    for direction in 0..directions {
        let wd = taken(slice_axis(w, 0, direction, 1), [4 * hidden, input_size])
        let rd = taken(slice_axis(r, 0, direction, 1), [4 * hidden, hidden])
        #: Every timestep's input half at once: it does not depend on the
        #: recurrence, so it is one product for the sequence rather than one
        #: for each step of it.
        var x_gates = jt.matmul(flat_x, wd, trans_b: true)
        wd.free()
        if bias is not null {
            #: `[directions, 8H]` is the input biases followed by the recurrent
            #: ones, and both land on the same gates.
            let row = taken(slice_axis(bias, 0, direction, 1), [8 * hidden])
            let wb = taken(slice_axis(row, 0, 0, 4 * hidden), [4 * hidden])
            let rb = taken(slice_axis(row, 0, 4 * hidden, 4 * hidden), [4 * hidden])
            let both = jt.add(wb, rb)
            let shifted = jt.add(x_gates, both)
            x_gates.free()
            x_gates = shifted
            row.free()
            wb.free()
            rb.free()
            both.free()
        }
        let given_h = maybe(ins, 5)
        let given_c = maybe(ins, 6)
        let start_h = if given_h is null {
            jt.Tensor.zeros([batch, hidden])
        } else {
            taken(slice_axis(given_h, 0, direction, 1), [batch, hidden])
        }
        let start_c = if given_c is null {
            jt.Tensor.zeros([batch, hidden])
        } else {
            taken(slice_axis(given_c, 0, direction, 1), [batch, hidden])
        }
        let (sequence, final_h, final_c) = lstm_direction(
            x_gates,
            rd,
            start_h,
            start_c,
            steps,
            batch,
            hidden,
            clip,
            facing == "reverse" or direction == 1
        )
        x_gates.free()
        rd.free()
        start_h.free()
        start_c.free()
        sequences.push(copy_of(sequence, [steps, 1, batch, hidden]))
        sequence.free()
        last_h.push(copy_of(final_h, [1, batch, hidden]))
        last_c.push(copy_of(final_c, [1, batch, hidden]))
        final_h.free()
        final_c.free()
    }
    flat_x.free()

    let all_y = if sequences.len() == 1 {
        sequences[0]
    } else {
        concat_along(sequences, 1, node.label())
    }
    let all_h = if last_h.len() == 1 { last_h[0] } else { concat_along(last_h, 0, node.label()) }
    let all_c = if last_c.len() == 1 { last_c[0] } else { concat_along(last_c, 0, node.label()) }
    if sequences.len() > 1 {
        for part in sequences { part.free() }
        for part in last_h { part.free() }
        for part in last_c { part.free() }
    }
    return [all_y, all_h, all_c]
}

#: One direction of a GRU, from `[seq, batch, input]` to `[seq, batch, H]`.
fn gru_direction(
    x_gates: Tensor,
    recurrent: Tensor,
    recurrent_h: Tensor,
    recurrent_bias: Tensor,
    initial_h: Tensor,
    steps: int,
    batch: int,
    hidden: int,
    clip: float,
    reverse: bool,
    linear_before_reset: bool
) -> tuple[Tensor, Tensor] {
    var state_h = initial_h
    var produced: list[Tensor] = []
    for index in 0..steps {
        let step = if reverse { steps - 1 - index } else { index }
        let recurred = jt.matmul(state_h, recurrent, trans_b: true)
        let offset = step * batch * 3 * hidden
        let next_h = jt.Tensor([batch, hidden])
        if linear_before_reset {
            kernel_for("gru_step_linear").dispatch_async(
                [
                    x_gates.buffer(),
                    recurred.buffer(),
                    recurrent_bias.buffer(),
                    state_h.buffer(),
                    next_h.buffer(),
                ],
                batch * hidden,
                [batch * hidden, hidden, f32_bits(clip), offset]
            )
        } else {
            #: The reset gate scales the hidden state before the weights see
            #: it, so this half stops to take a second product.
            let gate_z = jt.Tensor([batch, hidden])
            let scaled = jt.Tensor([batch, hidden])
            kernel_for("gru_reset").dispatch_async(
                [
                    x_gates.buffer(),
                    recurred.buffer(),
                    recurrent_bias.buffer(),
                    state_h.buffer(),
                    gate_z.buffer(),
                    scaled.buffer(),
                ],
                batch * hidden,
                [batch * hidden, hidden, f32_bits(clip), offset]
            )
            let reset_product = jt.matmul(scaled, recurrent_h, trans_b: true)
            kernel_for("gru_finish").dispatch_async(
                [
                    x_gates.buffer(),
                    reset_product.buffer(),
                    gate_z.buffer(),
                    state_h.buffer(),
                    next_h.buffer(),
                ],
                batch * hidden,
                [batch * hidden, hidden, f32_bits(clip), offset]
            )
            gate_z.free()
            scaled.free()
            reset_product.free()
        }
        recurred.free()
        state_h = next_h
        produced.push(next_h)
    }

    var ordered = produced
    if reverse {
        ordered = []
        for index in 0..produced.len() { ordered.push(produced[produced.len() - 1 - index]) }
    }
    var rows: list[Tensor] = []
    for part in ordered { rows.push(copy_of(part, [1, batch, hidden])) }
    let sequence = if rows.len() == 1 {
        rows[0]
    } else {
        concat_along(rows, 0, "a GRU's output sequence")
    }
    if rows.len() > 1 {
        for row in rows { row.free() }
    }
    let final_h = copy_of(state_h, [batch, hidden])
    for part in produced { part.free() }
    return (sequence, final_h)
}

#: ONNX `GRU`.
#:
#: Weights are `[directions, 3H, input]` and `[directions, 3H, H]` with the
#: gates ordered z, r, h, and the bias `[directions, 6H]` holds the input set
#: then the recurrent one. `linear_before_reset` decides whether the reset gate
#: scales the hidden state before or after the recurrent weights, which are
#: different answers -- and it decides where the recurrent bias for the
#: candidate gate belongs, since only one of the two puts it inside the reset.
pub fn op_gru(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let w = need(ins, 1, node)
    let r = need(ins, 2, node)
    let bias = maybe(ins, 3)
    if maybe(ins, 4) is not null {
        throw ValueError(
            f"{node.label()} gives per-sequence lengths, which this executor does not read; " +
                "every sequence in the batch has to be the same length"
        )
    }
    if node.attrs.get("activations", null) is not null {
        throw ValueError(
            f"{node.label()} names its own activations; this executor uses the sigmoid and " +
                "tanh the operator defaults to"
        )
    }
    if attr_int(node, "layout", 0) != 0 {
        throw ValueError(
            f"{node.label()} uses the batch-first layout, which this executor does not read"
        )
    }
    if x.rank() != 3 { throw ValueError(f"{node.label()} needs a 3-D input, got {x.shape}") }
    if w.rank() != 3 or r.rank() != 3 {
        throw ValueError(f"{node.label()} needs 3-D weights, got {w.shape} and {r.shape}")
    }
    let steps = x.shape[0]
    let batch = x.shape[1]
    let input_size = x.shape[2]
    let directions = w.shape[0]
    let hidden = attr_int(node, "hidden_size", r.shape[2])
    let facing = attr_str(node, "direction", "forward")
    let wanted = if facing == "bidirectional" { 2 } else { 1 }
    if directions != wanted or r.shape[0] != directions {
        throw ValueError(
            f"{node.label()} is {facing} but carries {directions} directions of weights"
        )
    }
    if w.shape[1] != 3 * hidden or r.shape[1] != 3 * hidden or r.shape[2] != hidden {
        throw ValueError(
            f"{node.label()} has weights {w.shape} and {r.shape} for {hidden} hidden units"
        )
    }
    if w.shape[2] != input_size {
        throw ValueError(f"{node.label()} has weights {w.shape} for {input_size} inputs")
    }
    let clip = attr_float(node, "clip", 0.0)
    let linear = attr_int(node, "linear_before_reset", 0) != 0

    let flat_x = copy_of(x, [steps * batch, input_size])
    var sequences: list[Tensor] = []
    var last_h: list[Tensor] = []
    for direction in 0..directions {
        let wd = taken(slice_axis(w, 0, direction, 1), [3 * hidden, input_size])
        let rd = taken(slice_axis(r, 0, direction, 1), [3 * hidden, hidden])
        let rh = taken(slice_axis(rd, 0, 2 * hidden, hidden), [hidden, hidden])
        var x_gates = jt.matmul(flat_x, wd, trans_b: true)
        wd.free()
        #: Always a real vector, zeros when the node carries none, so the
        #: kernels have one shape of input rather than two.
        var recurrent_bias = jt.Tensor.zeros([3 * hidden])
        if bias is not null {
            let row = taken(slice_axis(bias, 0, direction, 1), [6 * hidden])
            let wb = taken(slice_axis(row, 0, 0, 3 * hidden), [3 * hidden])
            let rb = taken(slice_axis(row, 0, 3 * hidden, 3 * hidden), [3 * hidden])
            #: Without `linear_before_reset` the candidate's recurrent bias
            #: falls outside the reset gate, which makes it a constant like the
            #: input biases -- so it moves to the input half, and the recurrent
            #: half is left carrying only the update and reset gates.
            var input_bias = wb
            var carried = rb
            if not linear {
                let head = taken(slice_axis(rb, 0, 0, 2 * hidden), [2 * hidden])
                let tail = taken(slice_axis(rb, 0, 2 * hidden, hidden), [hidden])
                let pad_tail = jt.Tensor.zeros([hidden])
                let pad_head = jt.Tensor.zeros([2 * hidden])
                carried = concat_along([head, pad_tail], 0, node.label())
                let widened = concat_along([pad_head, tail], 0, node.label())
                input_bias = jt.add(wb, widened)
                head.free()
                tail.free()
                pad_tail.free()
                pad_head.free()
                widened.free()
            }
            let shifted = jt.add(x_gates, input_bias)
            x_gates.free()
            x_gates = shifted
            recurrent_bias.free()
            recurrent_bias = carried
            if input_bias is not wb { input_bias.free() }
            wb.free()
            if carried is not rb { rb.free() }
            row.free()
        }
        let given_h = maybe(ins, 5)
        let start_h = if given_h is null {
            jt.Tensor.zeros([batch, hidden])
        } else {
            taken(slice_axis(given_h, 0, direction, 1), [batch, hidden])
        }
        let (sequence, final_h) = gru_direction(
            x_gates,
            rd,
            rh,
            recurrent_bias,
            start_h,
            steps,
            batch,
            hidden,
            clip,
            facing == "reverse" or direction == 1,
            linear
        )
        x_gates.free()
        rd.free()
        rh.free()
        start_h.free()
        recurrent_bias.free()
        sequences.push(copy_of(sequence, [steps, 1, batch, hidden]))
        sequence.free()
        last_h.push(copy_of(final_h, [1, batch, hidden]))
        final_h.free()
    }
    flat_x.free()

    let all_y = if sequences.len() == 1 {
        sequences[0]
    } else {
        concat_along(sequences, 1, node.label())
    }
    let all_h = if last_h.len() == 1 { last_h[0] } else { concat_along(last_h, 0, node.label()) }
    if sequences.len() > 1 {
        for part in sequences { part.free() }
        for part in last_h { part.free() }
    }
    return [all_y, all_h]
}

#: `copy_of` on a freshly cut slice, freeing the slice. Slicing allocates, and
#: every use of it here wants the copy and not the cut.
fn taken(cut: Tensor, shape: list[int]) -> Tensor {
    let kept = copy_of(cut, shape)
    cut.free()
    return kept
}
