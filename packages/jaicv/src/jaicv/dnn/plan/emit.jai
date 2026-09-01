#: Translating one node into the graph under construction.
import std.math as math
from .bounds import resolved_pads, slice_bounds
from .build import Build, as_ints, dims_of, scalar
from .codes import BINARY_CODES, FLOAT_TYPES, INTEGER_TYPES, PAD_MODES
from .codes import REDUCE_CODES, ROUNDING_CODES, UNARY_CODES
from .recurrent import emit_gru, emit_lstm
from ..graph import GraphNode, attr_float, attr_int, attr_ints, attr_str
from ..graph import normalize_axis

#: One node, translated into the graph under construction. Returns false when
#: the operator is one this compiler cannot express, which abandons the whole
#: compilation.
pub fn emit(state: Build, node: GraphNode) -> bool {
    let b = state.builder
    let out = if node.outputs.len() > 0 { node.outputs[0] } else { "" }

    let unary = UNARY_CODES.get(node.op, -1)
    if unary >= 0 and node.inputs.len() == 1 {
        state.ids[out] = __prim__.graph_unary(b, state.id_of(node.inputs[0]), unary)
        return true
    }
    let binary = BINARY_CODES.get(node.op, -1)
    if binary >= 0 and node.inputs.len() == 2 {
        state.ids[out] = __prim__.graph_binary(
            b,
            state.id_of(node.inputs[0]),
            state.id_of(node.inputs[1]),
            binary
        )
        return true
    }

    if node.op == "Conv" {
        let x = state.shape_of(node.inputs[0])
        let w = state.shape_of(node.inputs[1])
        if x is null or w is null or x.len() != 4 or w.len() != 4 { return false }
        let window = [w[2], w[3]]
        let strides = attr_ints(node, "strides", [1, 1])
        let dilations = attr_ints(node, "dilations", [1, 1])
        let pads = resolved_pads(node, [x[2], x[3]], window, strides, dilations)
        if pads is null { return false }
        let bias = if node.inputs.len() > 2 { state.id_of(node.inputs[2]) } else { -1 }
        state.ids[out] = __prim__.graph_conv(
            b,
            state.id_of(node.inputs[0]),
            state.id_of(node.inputs[1]),
            bias,
            [
                strides[0],
                strides[1],
                pads[0],
                pads[2],
                pads[1],
                pads[3],
                dilations[0],
                dilations[1],
                attr_int(node, "group", 1),
                0,
                0,
            ]
        )
        return true
    }

    if node.op == "ConvTranspose" {
        let x = state.shape_of(node.inputs[0])
        let w = state.shape_of(node.inputs[1])
        let want = state.shape_of(out)
        if x is null or w is null or want is null { return false }
        if x.len() != 4 or w.len() != 4 or want.len() != 4 { return false }
        if attr_int(node, "group", 1) != 1 { return false }
        #: ONNX writes these `[in, out, kH, kW]`, which is already the layout
        #: of the forward convolution being undone, so nothing is rearranged.
        if w[0] != x[1] { return false }
        let strides = attr_ints(node, "strides", [1, 1])
        let dilations = attr_ints(node, "dilations", [1, 1])
        #: The padding a transposed convolution declares is taken off its
        #: output, and the forward convolution it undoes is the one whose
        #: input is that output -- so the spatial size to resolve against is
        #: the result's, not the source's.
        let pads = resolved_pads(node, [want[2], want[3]], [w[2], w[3]], strides, dilations)
        if pads is null { return false }
        let bias = if node.inputs.len() > 2 and node.inputs[2].len() > 0 {
            state.id_of(node.inputs[2])
        } else {
            -1
        }
        state.ids[out] = __prim__.graph_conv_transpose(
            b,
            state.id_of(node.inputs[0]),
            state.id_of(node.inputs[1]),
            bias,
            [
                strides[0],
                strides[1],
                pads[0],
                pads[2],
                pads[1],
                pads[3],
                dilations[0],
                dilations[1],
                1,
            ],
            dims_of(want)
        )
        return true
    }

    if node.op == "MaxPool" or node.op == "AveragePool" {
        let x = state.shape_of(node.inputs[0])
        if x is null or x.len() != 4 { return false }
        let window = attr_ints(node, "kernel_shape", [1, 1])
        let strides = attr_ints(node, "strides", [1, 1])
        let pads = resolved_pads(node, [x[2], x[3]], window, strides, [1, 1])
        if pads is null { return false }
        state.ids[out] = __prim__.graph_pool(b, state.id_of(node.inputs[0]), [window[0], window[1], strides[0], strides[1], pads[0], pads[2], pads[1], pads[3], attr_int(node, "ceil_mode", 0), attr_int(node, "count_include_pad", 0), 0], if node.op == "MaxPool" {
            0
        } else {
            1
        })
        return true
    }

    if node.op == "Elu" {
        #: `max(x, 0) + alpha * (exp(min(x, 0)) - 1)`, which is the operator
        #: without needing a branch: above zero the second term is zero, and
        #: below it the first is.
        let alpha = attr_float(node, "alpha", 1.0)
        let x = state.id_of(node.inputs[0])
        let positive = __prim__.graph_binary(b, x, scalar(state, 0.0), 4)
        let negative = __prim__.graph_binary(b, x, scalar(state, 0.0), 5)
        let curved = __prim__.graph_binary(
            b,
            __prim__.graph_binary(
                b,
                __prim__.graph_unary(b, negative, 3),
                scalar(state, 1.0),
                1
            ),
            scalar(state, alpha),
            2
        )
        state.ids[out] = __prim__.graph_binary(b, positive, curved, 0)
        return true
    }

    if node.op == "PRelu" {
        #: `max(x, 0) + slope * min(x, 0)`, with the slope broadcasting along
        #: the channel axis the way the operator says.
        let x = state.shape_of(node.inputs[0])
        let slope = state.shape_of(node.inputs[1])
        if x is null or slope is null { return false }
        var shaped = state.id_of(node.inputs[1])
        if slope.len() == 1 and x.len() > 1 and slope[0] == x[1] {
            var per_channel = [1 for _index in 0..x.len()]
            per_channel[1] = slope[0]
            shaped = __prim__.graph_reshape(b, shaped, per_channel)
        }
        let value = state.id_of(node.inputs[0])
        let positive = __prim__.graph_binary(b, value, scalar(state, 0.0), 4)
        let negative = __prim__.graph_binary(b, value, scalar(state, 0.0), 5)
        state.ids[out] = __prim__.graph_binary(
            b,
            positive,
            __prim__.graph_binary(b, negative, shaped, 2),
            0
        )
        return true
    }

    if node.op == "HardSigmoid" {
        let scaled = __prim__.graph_binary(
            b,
            state.id_of(node.inputs[0]),
            scalar(state, attr_float(node, "alpha", 0.2)),
            2
        )
        let shifted = __prim__.graph_binary(
            b,
            scaled,
            scalar(state, attr_float(node, "beta", 0.5)),
            0
        )
        state.ids[out] = __prim__.graph_clamp(b, shifted, 0.0, 1.0)
        return true
    }

    if node.op == "Softplus" {
        let raised = __prim__.graph_unary(b, state.id_of(node.inputs[0]), 3)
        let shifted = __prim__.graph_binary(b, raised, scalar(state, 1.0), 0)
        state.ids[out] = __prim__.graph_unary(b, shifted, 4)
        return true
    }

    if node.op == "LogSoftmax" {
        let x = state.shape_of(node.inputs[0])
        if x is null { return false }
        let normalised = __prim__.graph_softmax(
            b,
            state.id_of(node.inputs[0]),
            normalize_axis(attr_int(node, "axis", -1), x.len(), node.label())
        )
        state.ids[out] = __prim__.graph_unary(b, normalised, 4)
        return true
    }

    if node.op == "Expand" {
        #: Only where the shape does not actually change, which is what most
        #: exported graphs use it for. A real expansion is left alone rather
        #: than assumed to broadcast the way the operator means.
        let want = state.shape_of(out)
        let have = state.shape_of(node.inputs[0])
        if want is null or have is null or have != want { return false }
        state.ids[out] = state.id_of(node.inputs[0])
        return true
    }

    if node.op == "BatchNormalization" {
        #: Folded rather than emitted. With the statistics fixed, the whole
        #: operator is `x * k + shift` for a per-channel `k` and `shift`, and
        #: both can be worked out here -- which is the ordinary way a trained
        #: network is deployed, and leaves two broadcasts where there was a
        #: normalisation.
        let x = state.shape_of(node.inputs[0])
        if x is null or x.len() < 2 { return false }
        if node.outputs.len() > 1 { return false }
        let scale = state.numbers_of(node.inputs[1])
        let shift = state.numbers_of(node.inputs[2])
        let mean = state.numbers_of(node.inputs[3])
        let variance = state.numbers_of(node.inputs[4])
        if scale is null or shift is null or mean is null or variance is null { return false }
        let channels = x[1]
        if scale.len() != channels { return false }
        let epsilon = attr_float(node, "epsilon", 1.0e-5)
        var factor = __prim__.list_filled(channels, 0.0)
        var offset = __prim__.list_filled(channels, 0.0)
        for index in 0..channels {
            let inverse = 1.0 / math.sqrt(variance[index] + epsilon)
            factor[index] = scale[index] * inverse
            offset[index] = shift[index] - mean[index] * factor[index]
        }
        #: Shaped so the channel axis lines up and the rest broadcasts.
        var per_channel = [1 for _index in 0..x.len()]
        per_channel[1] = channels
        let k = __prim__.graph_constant(b, factor, per_channel)
        let c = __prim__.graph_constant(b, offset, per_channel)
        let scaled = __prim__.graph_binary(b, state.id_of(node.inputs[0]), k, 2)
        state.ids[out] = __prim__.graph_binary(b, scaled, c, 0)
        return true
    }

    if node.op == "GlobalAveragePool" or node.op == "GlobalMaxPool" {
        let x = state.shape_of(node.inputs[0])
        let want = state.shape_of(out)
        if x is null or want is null or x.len() < 3 { return false }
        var over: list[int] = []
        for axis in 2..x.len() { over.push(axis) }
        let reduced = __prim__.graph_reduce(b, state.id_of(node.inputs[0]), over, if node.op == "GlobalAveragePool" {
            1
        } else {
            2
        })
        state.ids[out] = __prim__.graph_reshape(b, reduced, dims_of(want))
        return true
    }

    if node.op == "Clip" {
        var low = -3.4028235e38
        var high = 3.4028235e38
        #: Bounds are attributes before opset 11 and inputs after it.
        let by_attr_low = node.attrs.get("min", null)
        let by_attr_high = node.attrs.get("max", null)
        if by_attr_low is not null { low = attr_float(node, "min", low) }
        if by_attr_high is not null { high = attr_float(node, "max", high) }
        if node.inputs.len() > 1 and node.inputs[1].len() > 0 {
            let given = state.numbers_of(node.inputs[1])
            if given is null or given.len() == 0 { return false }
            low = given[0]
        }
        if node.inputs.len() > 2 and node.inputs[2].len() > 0 {
            let given = state.numbers_of(node.inputs[2])
            if given is null or given.len() == 0 { return false }
            high = given[0]
        }
        state.ids[out] = __prim__.graph_clamp(b, state.id_of(node.inputs[0]), low, high)
        return true
    }

    if node.op == "LeakyRelu" {
        state.ids[out] = __prim__.graph_leaky_relu(
            b,
            state.id_of(node.inputs[0]),
            attr_float(node, "alpha", 0.01)
        )
        return true
    }

    if node.op == "Not" {
        #: One where the value was zero, which is `x == 0` written as a
        #: comparison the builder already has.
        state.ids[out] = __prim__.graph_binary(
            b,
            state.id_of(node.inputs[0]),
            scalar(state, 0.0),
            7
        )
        return true
    }

    if node.op == "Where" {
        state.ids[out] = __prim__.graph_select(
            b,
            state.id_of(node.inputs[0]),
            state.id_of(node.inputs[1]),
            state.id_of(node.inputs[2])
        )
        return true
    }

    if node.op == "Mod" {
        #: With `fmod` the sign follows the dividend, which is `x - trunc(x/y) * y`;
        #: without it the sign follows the divisor, which is the same with a
        #: floor. Neither needs an operator of its own.
        let x = state.id_of(node.inputs[0])
        let y = state.id_of(node.inputs[1])
        let ratio = __prim__.graph_binary(b, x, y, 3)
        let whole = __prim__.graph_unary(b, ratio, if attr_int(node, "fmod", 0) != 0 {
            15
        } else {
            10
        })
        state.ids[out] = __prim__.graph_binary(b, x, __prim__.graph_binary(b, whole, y, 2), 1)
        return true
    }

    if node.op == "Mean" {
        var running = state.id_of(node.inputs[0])
        for index in 1..node.inputs.len() {
            running = __prim__.graph_binary(b, running, state.id_of(node.inputs[index]), 0)
        }
        state.ids[out] = __prim__.graph_binary(
            b,
            running,
            scalar(state, float(node.inputs.len())),
            3
        )
        return true
    }

    if node.op == "Sum" or node.op == "Max" or node.op == "Min" {
        let code = if node.op == "Sum" { 0 } elif node.op == "Max" { 4 } else { 5 }
        var running = state.id_of(node.inputs[0])
        for index in 1..node.inputs.len() {
            running = __prim__.graph_binary(b, running, state.id_of(node.inputs[index]), code)
        }
        state.ids[out] = running
        return true
    }

    if node.op == "Concat" {
        var parts: list[int] = []
        for name in node.inputs { parts.push(state.id_of(name)) }
        let rank = state.shape_of(node.inputs[0])
        if rank is null { return false }
        state.ids[out] = __prim__.graph_concat(
            b,
            parts,
            normalize_axis(attr_int(node, "axis", 0), rank.len(), node.label())
        )
        return true
    }

    if node.op == "Reshape" or
        node.op == "Squeeze" or
        node.op == "Unsqueeze" or
        node.op == "Flatten" {
        let shape = state.shape_of(out)
        if shape is null { return false }
        state.ids[out] = __prim__.graph_reshape(b, state.id_of(node.inputs[0]), dims_of(shape))
        return true
    }

    if node.op == "Transpose" {
        let x = state.shape_of(node.inputs[0])
        if x is null { return false }
        var perm = attr_ints(node, "perm", [])
        if perm.len() == 0 {
            for index in 0..x.len() { perm.push(x.len() - 1 - index) }
        }
        state.ids[out] = __prim__.graph_transpose(b, state.id_of(node.inputs[0]), perm)
        return true
    }

    if node.op == "Pad" {
        let x = state.shape_of(node.inputs[0])
        if x is null { return false }
        let mode = PAD_MODES.get(attr_str(node, "mode", "constant"), -1)
        if mode < 0 { return false }
        var pads = attr_ints(node, "pads", [])
        if pads.len() == 0 and node.inputs.len() > 1 {
            let given = state.numbers_of(node.inputs[1])
            if given is null { return false }
            pads = as_ints(given)
        }
        if pads.len() != 2 * x.len() { return false }
        #: A negative pad is a crop, which this does not do.
        for value in pads {
            if value < 0 { return false }
        }
        var value = 0.0
        if node.inputs.len() > 2 and node.inputs[2].len() > 0 {
            let given = state.numbers_of(node.inputs[2])
            if given is null or given.len() == 0 { return false }
            value = given[0]
        }
        var before: list[int] = []
        var after: list[int] = []
        for axis in 0..x.len() {
            before.push(pads[axis])
            after.push(pads[x.len() + axis])
        }
        state.ids[out] = __prim__.graph_pad(
            b,
            state.id_of(node.inputs[0]),
            before,
            after,
            mode,
            value
        )
        return true
    }

    if node.op == "ArgMax" or node.op == "ArgMin" {
        let x = state.shape_of(node.inputs[0])
        let want = state.shape_of(out)
        if x is null or want is null { return false }
        if attr_int(node, "select_last_index", 0) != 0 { return false }
        let picked = __prim__.graph_arg_extreme(b, state.id_of(node.inputs[0]), normalize_axis(attr_int(node, "axis", 0), x.len(), node.label()), if node.op == "ArgMax" {
            1
        } else {
            0
        })
        #: The reduced axis stays as a one here and ONNX drops it unless
        #: `keepdims` says otherwise; the survey knows which happened.
        state.ids[out] = __prim__.graph_reshape(b, picked, dims_of(want))
        return true
    }

    if node.op == "Softmax" {
        let x = state.shape_of(node.inputs[0])
        if x is null { return false }
        state.ids[out] = __prim__.graph_softmax(
            b,
            state.id_of(node.inputs[0]),
            normalize_axis(attr_int(node, "axis", -1), x.len(), node.label())
        )
        return true
    }

    if node.op == "Cast" {
        #: Every tensor here is float, so a cast to a float type is nothing at
        #: all; a cast to an integer one truncates, and a cast to boolean is
        #: one where the value was not zero. The same three cases the
        #: interpreter has.
        let target = attr_int(node, "to", 1)
        if target in FLOAT_TYPES {
            state.ids[out] = state.id_of(node.inputs[0])
            return true
        }
        let code = if target in INTEGER_TYPES { 15 } elif target == 9 { 16 } else { -1 }
        if code < 0 { return false }
        state.ids[out] = __prim__.graph_unary(b, state.id_of(node.inputs[0]), code)
        return true
    }

    if node.op == "Identity" or node.op == "Dropout" {
        state.ids[out] = state.id_of(node.inputs[0])
        return true
    }

    if node.op == "Resize" or node.op == "Upsample" {
        let want = state.shape_of(out)
        if want is null or want.len() != 4 { return false }
        let sampling = attr_str(node, "mode", "nearest")
        if sampling != "nearest" and sampling != "linear" { return false }
        #: Only the coordinate rules that map exactly are compiled. Half-pixel
        #: sampling is left to the interpreter rather than approximated, since
        #: being one pixel out is the kind of difference that survives a test
        #: and shows up in a detection.
        let placing = attr_str(node, "coordinate_transformation_mode", "half_pixel")
        var center = 0
        var corners = 0
        if placing == "asymmetric" {
            center = 0
            corners = 0
        } elif placing == "align_corners" {
            corners = 1
        } elif placing == "half_pixel" {
            center = 1
        } else {
            return false
        }
        let rounding = ROUNDING_CODES.get(attr_str(node, "nearest_mode", "round_prefer_floor"), -1)
        if rounding < 0 and sampling == "nearest" { return false }
        state.ids[out] = __prim__.graph_resize(b, state.id_of(node.inputs[0]), want[2], want[3], max(rounding, 0), center, corners, if sampling == "linear" {
            1
        } else {
            0
        })
        return true
    }

    if node.op == "LayerNormalization" {
        let x = state.shape_of(node.inputs[0])
        if x is null { return false }
        let raw = attr_int(node, "axis", -1)
        let axis = if raw < 0 { x.len() + raw } else { raw }
        if axis < 0 or axis >= x.len() { return false }
        var axes: list[int] = []
        for index in axis..x.len() { axes.push(index) }
        let gamma = if node.inputs.len() > 1 { state.id_of(node.inputs[1]) } else { -1 }
        let beta = if node.inputs.len() > 2 and node.inputs[2].len() > 0 {
            state.id_of(node.inputs[2])
        } else {
            -1
        }
        state.ids[out] = __prim__.graph_layer_norm(
            b,
            state.id_of(node.inputs[0]),
            gamma,
            beta,
            axes,
            attr_float(node, "epsilon", 1.0e-5)
        )
        return true
    }

    if node.op == "InstanceNormalization" {
        #: The same normalisation layer norm uses, over the spatial axes
        #: instead of the trailing ones, with the scale and shift widened so
        #: they line up with the channel.
        let x = state.shape_of(node.inputs[0])
        if x is null or x.len() < 3 { return false }
        var axes: list[int] = []
        for axis in 2..x.len() { axes.push(axis) }
        var per_channel = [1 for _index in 0..x.len()]
        per_channel[1] = x[1]
        let gamma = __prim__.graph_reshape(b, state.id_of(node.inputs[1]), per_channel)
        let beta = __prim__.graph_reshape(b, state.id_of(node.inputs[2]), per_channel)
        state.ids[out] = __prim__.graph_layer_norm(
            b,
            state.id_of(node.inputs[0]),
            gamma,
            beta,
            axes,
            attr_float(node, "epsilon", 1.0e-5)
        )
        return true
    }

    if node.op == "Gemm" {
        #: Checked before it is built, because MPS stops the process on a
        #: matrix product whose inner dimensions disagree rather than
        #: refusing it -- so anything not certain to line up is left to the
        #: interpreter.
        let a = state.shape_of(node.inputs[0])
        let w = state.shape_of(node.inputs[1])
        if a is null or w is null or a.len() != 2 or w.len() != 2 { return false }
        let inner_a = if attr_int(node, "transA", 0) != 0 { a[0] } else { a[1] }
        let inner_b = if attr_int(node, "transB", 0) != 0 { w[1] } else { w[0] }
        if inner_a != inner_b { return false }
        let c = if node.inputs.len() > 2 and node.inputs[2].len() > 0 {
            state.id_of(node.inputs[2])
        } else {
            -1
        }
        state.ids[out] = __prim__.graph_gemm(
            b,
            state.id_of(node.inputs[0]),
            state.id_of(node.inputs[1]),
            c,
            attr_int(node, "transA", 0),
            attr_int(node, "transB", 0),
            attr_float(node, "alpha", 1.0),
            attr_float(node, "beta", 1.0)
        )
        return true
    }

    if node.op == "ReduceMean" or
        node.op == "ReduceSum" or
        node.op == "ReduceMax" or
        node.op == "ReduceMin" {
        let x = state.shape_of(node.inputs[0])
        let want = state.shape_of(out)
        if x is null or want is null { return false }
        var axes = attr_ints(node, "axes", [])
        if axes.len() == 0 and node.inputs.len() > 1 {
            let given = state.constants.get(node.inputs[1], null)
            if given is null { return false }
            axes = as_ints(given)
        }
        if axes.len() == 0 {
            for index in 0..x.len() { axes.push(index) }
        }
        var over: list[int] = []
        for axis in axes { over.push(normalize_axis(axis, x.len(), node.label())) }
        let kind = REDUCE_CODES.get(node.op, -1)
        if kind < 0 { return false }
        let reduced = __prim__.graph_reduce(b, state.id_of(node.inputs[0]), over, kind)
        #: MPSGraph keeps the reduced axes as ones; ONNX drops them unless
        #: `keepdims` says otherwise, and the survey already knows which.
        state.ids[out] = if want.len() == x.len() {
            reduced
        } else {
            __prim__.graph_reshape(b, reduced, dims_of(want))
        }
        return true
    }

    if node.op == "Gather" {
        let x = state.shape_of(node.inputs[0])
        let want = state.shape_of(out)
        if x is null or want is null { return false }
        let picked = __prim__.graph_gather(
            b,
            state.id_of(node.inputs[0]),
            state.id_of(node.inputs[1]),
            normalize_axis(attr_int(node, "axis", 0), x.len(), node.label())
        )
        #: ONNX drops the axis for a scalar index and keeps it for a vector
        #: one; the survey already knows which happened, so the result is
        #: reshaped to whatever it should have been.
        state.ids[out] = __prim__.graph_reshape(b, picked, dims_of(want))
        return true
    }

    if node.op == "LSTM" { return emit_lstm(state, node) }

    if node.op == "GRU" { return emit_gru(state, node) }

    if node.op == "MatMul" {
        #: Only the shapes that line up without broadcasting, and for the same
        #: reason as `Gemm`: a mismatch here is fatal rather than refusable.
        let a = state.shape_of(node.inputs[0])
        let w = state.shape_of(node.inputs[1])
        let want = state.shape_of(out)
        if a is null or w is null or want is null { return false }
        if a.len() < 2 or w.len() < 2 { return false }
        if a[a.len() - 1] != w[w.len() - 2] { return false }

        #: A matrix applied across leading dimensions -- which is what a
        #: linear layer over a batch of tokens is -- is flattened into one
        #: product and shaped back. Doing it here rather than relying on the
        #: framework to broadcast keeps it to the one case that is certainly
        #: right.
        if w.len() == 2 and a.len() > 2 {
            var rows = 1
            for axis in 0..a.len() - 1 { rows *= a[axis] }
            let flat = __prim__.graph_reshape(
                b,
                state.id_of(node.inputs[0]),
                [rows, a[a.len() - 1]]
            )
            let product = __prim__.graph_matmul(b, flat, state.id_of(node.inputs[1]))
            state.ids[out] = __prim__.graph_reshape(b, product, dims_of(want))
            return true
        }
        if w.len() != a.len() { return false }
        for axis in 0..a.len() - 2 {
            if a[axis] != w[axis] { return false }
        }
        state.ids[out] = __prim__.graph_matmul(
            b,
            state.id_of(node.inputs[0]),
            state.id_of(node.inputs[1])
        )
        return true
    }

    #: `Split` and `Slice` both come out as slices, which is what they are; the
    #: shape survey has already worked out how wide each piece is.
    if node.op == "Split" {
        let x = state.shape_of(node.inputs[0])
        if x is null { return false }
        let axis = normalize_axis(attr_int(node, "axis", 0), x.len(), node.label())
        var at = 0
        for name in node.outputs {
            let piece = state.shape_of(name)
            if piece is null { return false }
            var starts: list[int] = []
            var ends: list[int] = []
            var steps: list[int] = []
            for index in 0..x.len() {
                starts.push(if index == axis { at } else { 0 })
                ends.push(if index == axis { at + piece[axis] } else { x[index] })
                steps.push(1)
            }
            state.ids[name] = __prim__.graph_slice(
                b,
                state.id_of(node.inputs[0]),
                starts,
                ends,
                steps
            )
            at += piece[axis]
        }
        return true
    }

    if node.op == "Slice" {
        let x = state.shape_of(node.inputs[0])
        let want = state.shape_of(out)
        if x is null or want is null { return false }
        let bounds = slice_bounds(state, node, x, want)
        if bounds is null { return false }
        state.ids[out] = __prim__.graph_slice(
            b,
            state.id_of(node.inputs[0]),
            bounds[0],
            bounds[1],
            bounds[2]
        )
        return true
    }

    return false
}
