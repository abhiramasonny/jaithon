#!/usr/bin/env python3
"""Record what a reference runtime does, for `tests/test_dnn_graph.jai`.

Every operator case is run through onnxruntime, which is the definition of
what an ONNX operator means, rather than through a hand-reading of the spec.
The end-to-end network is run through torch instead, because that is what
someone porting a model actually has in front of them.

The two halves need different packages, so they run under different
interpreters:

    cd /tmp && uv run --with onnx --with onnxruntime --with numpy python \\
        ~/Developer/projects/jaithon/packages/jaicv/tools/dnn_graph_oracle.py \\
        ops ~/Developer/projects/jaithon/packages/jaicv/tests/oracle/dnn_graph_cases.txt

    ~/.venvs/scratch/bin/python packages/jaicv/tools/dnn_graph_oracle.py \\
        e2e packages/jaicv/tests/oracle/dnn_graph_network.txt

Both files are checked in, so the test runs without either package installed.
"""
from __future__ import annotations

import sys

import numpy as np


# ---------------------------------------------------------------- formatting


def numbers(values) -> str:
    out = []
    for value in np.asarray(values).reshape(-1).astype(np.float64):
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("an expected value came out infinite or NaN")
        out.append(repr(value))
    return " ".join(out)


def tensor_line(tag: str, array) -> str:
    array = np.asarray(array)
    shape = list(array.shape) or [1]
    dims = " ".join(str(int(d)) for d in shape)
    return f"{tag} {len(shape)} {dims} {numbers(array)}".rstrip()


class TensorAttr:
    """An attribute that is itself a tensor, such as `Constant`'s `value`."""

    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)


def attr_line(key, value) -> str:
    if isinstance(value, TensorAttr):
        return f"attr {key} " + tensor_line("tensor", value.array)
    if isinstance(value, bool):
        return f"attr {key} int {int(value)}"
    if isinstance(value, str):
        return f"attr {key} str {value}"
    if isinstance(value, (int, np.integer)):
        return f"attr {key} int {int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"attr {key} float {repr(float(value))}"
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in value):
            return f"attr {key} ints " + " ".join(str(int(v)) for v in value)
        return f"attr {key} floats " + " ".join(repr(float(v)) for v in value)
    raise TypeError(f"no attribute line for {value!r}")


# --------------------------------------------------------------- the cases

CASES: list[dict] = []


def case(group, name, op, inputs, attrs=None, outputs=1, opset=13):
    CASES.append(
        {
            "group": group,
            "name": name,
            "op": op,
            "inputs": list(inputs),
            "attrs": dict(attrs or {}),
            "outputs": outputs,
            "opset": opset,
        }
    )


def spread(*shape, low=-2.3, high=2.6, seed=0):
    """Deterministic values that avoid zero, so Div and Log stay finite."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(low, high, size=shape).astype(np.float32)
    values[np.abs(values) < 0.05] += 0.37
    return np.ascontiguousarray(values)


def build_cases():
    unary = np.array([[-2.5, -1.0, -0.3, 0.125], [0.4, 1.2, 2.7, 3.5]], dtype=np.float32)
    positive = np.abs(unary) + 0.25

    for op in ["Relu", "Sigmoid", "Tanh", "Exp", "Abs", "Neg", "Erf", "Softplus"]:
        case("elementwise", f"{op.lower()}_basic", op, [unary])
    case("elementwise", "sqrt_basic", "Sqrt", [positive])
    case("elementwise", "log_basic", "Log", [positive])
    case("elementwise", "elu_alpha", "Elu", [unary], {"alpha": 0.7})
    case("elementwise", "leakyrelu_alpha", "LeakyRelu", [unary], {"alpha": 0.13})
    case("elementwise", "leakyrelu_default", "LeakyRelu", [unary])
    case("elementwise", "hardsigmoid", "HardSigmoid", [unary], {"alpha": 0.15, "beta": 0.4})
    case("elementwise", "clip_attrs", "Clip", [unary], {"min": -0.5, "max": 1.5}, opset=6)
    case(
        "elementwise",
        "clip_inputs",
        "Clip",
        [unary, np.float32(-0.5), np.float32(1.5)],
    )
    case("elementwise", "clip_only_min", "Clip", [unary, np.float32(0.2), None])
    case(
        "elementwise",
        "prelu_channel",
        "PRelu",
        [spread(1, 3, 2, 2, seed=1), np.array([0.1, -0.25, 0.5], dtype=np.float32).reshape(3, 1, 1)],
    )
    case("elementwise", "not_basic", "Not", [np.array([[1, 0], [0, 1]], dtype=bool)])
    case("elementwise", "softmax_last", "Softmax", [spread(2, 3, 4, seed=2)])
    case("elementwise", "softmax_axis1", "Softmax", [spread(2, 3, 4, seed=2)], {"axis": 1})
    case("elementwise", "logsoftmax_last", "LogSoftmax", [spread(2, 3, 4, seed=2)])
    case("elementwise", "logsoftmax_axis0", "LogSoftmax", [spread(2, 3, 4, seed=2)], {"axis": 0})

    # Broadcasting, once for every elementwise operator that takes two sides.
    left = spread(2, 3, 4, seed=3)
    right = spread(1, 3, 1, seed=4)
    row = spread(4, seed=5)
    for op in ["Add", "Sub", "Mul", "Div", "Pow", "Min", "Max", "Sum", "Mean"]:
        base = np.abs(left) + 0.3 if op == "Pow" else left
        case("broadcast", f"{op.lower()}_broadcast", op, [base, right])
        case("broadcast", f"{op.lower()}_broadcast_row", op, [base, row])
        case("broadcast", f"{op.lower()}_same", op, [base, spread(2, 3, 4, seed=6)])
    for op in ["Equal", "Greater", "Less"]:
        case("broadcast", f"{op.lower()}_broadcast", op, [left, right])
    flags_a = (spread(2, 3, 4, seed=7) > 0).astype(bool)
    flags_b = (spread(1, 3, 1, seed=8) > 0).astype(bool)
    case("broadcast", "and_broadcast", "And", [flags_a, flags_b])
    case("broadcast", "or_broadcast", "Or", [flags_a, flags_b])
    case(
        "broadcast",
        "where_broadcast",
        "Where",
        [(spread(2, 1, 4, seed=9) > 0).astype(bool), spread(2, 3, 1, seed=10), spread(4, seed=11)],
    )
    case("broadcast", "sum_three", "Sum", [left, right, row])
    case("broadcast", "mean_three", "Mean", [left, right, row])

    # Shape and layout.
    data = spread(2, 3, 4, seed=12)
    case("shape", "reshape_infer", "Reshape", [data, np.array([2, -1], dtype=np.int64)])
    case("shape", "reshape_zero", "Reshape", [data, np.array([0, 12], dtype=np.int64)])
    case("shape", "flatten_axis2", "Flatten", [data], {"axis": 2})
    case("shape", "flatten_axis0", "Flatten", [data], {"axis": 0})
    case("shape", "flatten_negative", "Flatten", [data], {"axis": -1})
    case(
        "shape",
        "squeeze_axes",
        "Squeeze",
        [spread(1, 3, 1, 4, seed=13), np.array([0, 2], dtype=np.int64)],
    )
    case(
        "shape",
        "unsqueeze_axes",
        "Unsqueeze",
        [spread(3, 4, seed=14), np.array([0, 3], dtype=np.int64)],
    )
    case("shape", "transpose_default", "Transpose", [data])
    case("shape", "transpose_perm", "Transpose", [data], {"perm": [1, 0, 2]})
    case("shape", "transpose_perm4", "Transpose", [spread(2, 3, 4, 5, seed=15)], {"perm": [0, 3, 1, 2]})
    case(
        "shape",
        "slice_steps",
        "Slice",
        [
            data,
            np.array([0, 1], dtype=np.int64),
            np.array([2, 4], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
        ],
    )
    case(
        "shape",
        "slice_negative_step",
        "Slice",
        [
            data,
            np.array([3], dtype=np.int64),
            np.array([-5], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([-1], dtype=np.int64),
        ],
    )
    case(
        "shape",
        "slice_negative_step_two",
        "Slice",
        [
            data,
            np.array([3, 2], dtype=np.int64),
            np.array([-5, -4], dtype=np.int64),
            np.array([2, 1], dtype=np.int64),
            np.array([-2, -1], dtype=np.int64),
        ],
    )
    case(
        "shape",
        "slice_step_two_forward",
        "Slice",
        [
            data,
            np.array([0], dtype=np.int64),
            np.array([4], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([3], dtype=np.int64),
        ],
    )
    case(
        "shape",
        "slice_open_ended",
        "Slice",
        [data, np.array([1], dtype=np.int64), np.array([9223372036854775807], dtype=np.int64), np.array([1], dtype=np.int64)],
    )
    # A range that comes up empty. Torch's attention exports slice the tail off
    # a shape and get nothing whenever the tail is not there, then concatenate
    # the nothing straight back in, so the executor has to carry empties.
    case(
        "shape",
        "slice_empty",
        "Slice",
        [
            data,
            np.array([3], dtype=np.int64),
            np.array([9223372036854775807], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ],
    )
    case(
        "shape",
        "concat_with_an_empty",
        "Concat",
        [spread(2, 0, 4, seed=151), spread(2, 3, 4, seed=152)],
        {"axis": 1},
    )
    case(
        "shape",
        "gather_axis1",
        "Gather",
        [data, np.array([[0, 2], [1, 1]], dtype=np.int64)],
        {"axis": 1},
    )
    case("shape", "gather_axis0", "Gather", [data, np.array([1, 0], dtype=np.int64)])
    case("shape", "concat_axis1", "Concat", [data, spread(2, 5, 4, seed=16)], {"axis": 1})
    case("shape", "concat_axis_last", "Concat", [data, spread(2, 3, 2, seed=17)], {"axis": -1})
    case("shape", "split_uneven", "Split", [data, np.array([1, 2], dtype=np.int64)], {"axis": 1}, outputs=2)
    case("shape", "split_even", "Split", [data], {"axis": 2}, outputs=2, opset=11)
    padded = spread(1, 1, 3, 4, seed=18)
    for mode in ["constant", "reflect", "edge"]:
        case(
            "shape",
            f"pad_{mode}",
            "Pad",
            [padded, np.array([0, 0, 1, 2, 0, 0, 2, 1], dtype=np.int64)],
            {"mode": mode},
        )
    case(
        "shape",
        "pad_value",
        "Pad",
        [padded, np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64), np.float32(-3.5)],
    )
    case("shape", "shape_all", "Shape", [spread(2, 3, 4, 5, seed=19)], opset=15)
    case("shape", "shape_slice", "Shape", [spread(2, 3, 4, 5, seed=19)], {"start": 1, "end": 3}, opset=15)
    case(
        "shape",
        "constant_tensor",
        "Constant",
        [],
        {"value": TensorAttr(np.array([[1.5, -2.0], [0.25, 8.0]], dtype=np.float32))},
    )
    case(
        "shape",
        "constant_of_shape",
        "ConstantOfShape",
        [np.array([2, 3], dtype=np.int64)],
        {"value": TensorAttr(np.array([7.25], dtype=np.float32))},
    )
    case("shape", "cast_to_int", "Cast", [np.array([-2.7, -0.4, 0.6, 3.9], dtype=np.float32)], {"to": 6})
    case("shape", "cast_to_bool", "Cast", [np.array([-2.7, 0.0, 0.6, 0.0], dtype=np.float32)], {"to": 9})
    case("shape", "identity", "Identity", [data])
    case("shape", "dropout", "Dropout", [data], opset=12)

    case("elementwise", "clip_only_max", "Clip", [unary, None, np.float32(1.5)])
    case("elementwise", "softmax_rank2", "Softmax", [spread(3, 5, seed=61)])
    case("elementwise", "logsoftmax_rank2", "LogSoftmax", [spread(3, 5, seed=61)])
    case("broadcast", "sum_single", "Sum", [left])
    case(
        "broadcast",
        "where_scalars",
        "Where",
        [np.array([True], dtype=bool), spread(2, 3, seed=62), np.float32(0.5)],
    )
    case(
        "shape",
        "concat_three",
        "Concat",
        [data, spread(2, 1, 4, seed=63), spread(2, 2, 4, seed=64)],
        {"axis": 1},
    )
    case("shape", "split_three", "Split", [spread(2, 6, seed=65)], {"axis": 1}, outputs=3, opset=11)
    case("shape", "gather_negative_axis", "Gather", [data, np.array([3, 0, 1], dtype=np.int64)], {"axis": -1})
    case("shape", "reshape_grow", "Reshape", [data, np.array([2, 3, 2, 2], dtype=np.int64)])
    case(
        "shape",
        "slice_unordered_axes",
        "Slice",
        [
            data,
            np.array([1, 0], dtype=np.int64),
            np.array([4, 2], dtype=np.int64),
            np.array([2, 1], dtype=np.int64),
            np.array([2, 1], dtype=np.int64),
        ],
    )
    case(
        "shape",
        "pad_negative",
        "Pad",
        [padded, np.array([0, 0, -1, 1, 0, 0, 0, -2], dtype=np.int64)],
        {"mode": "constant"},
        opset=18,
    )
    case("linear", "matmul_batched_broadcast", "MatMul", [spread(2, 1, 3, 4, seed=66), spread(1, 5, 4, 2, seed=67)])

    # Convolution and pooling.
    image = spread(1, 2, 5, 5, seed=20)
    weights = spread(3, 2, 3, 3, seed=21)
    bias = spread(3, seed=22)
    case("window", "conv_pad1", "Conv", [image, weights, bias], {"pads": [1, 1, 1, 1]})
    case("window", "conv_valid", "Conv", [image, weights], {})
    case("window", "conv_stride2", "Conv", [image, weights, bias], {"strides": [2, 2], "pads": [1, 1, 1, 1]})
    case(
        "window",
        "conv_asymmetric_pad",
        "Conv",
        [image, weights, bias],
        {"pads": [0, 1, 2, 0]},
    )
    case("window", "conv_same_upper", "Conv", [image, weights, bias], {"auto_pad": "SAME_UPPER", "kernel_shape": [3, 3], "strides": [2, 2]})
    case(
        "window",
        "conv_grouped",
        "Conv",
        [spread(1, 4, 5, 5, seed=23), spread(6, 2, 3, 3, seed=24), spread(6, seed=25)],
        {"group": 2, "pads": [1, 1, 1, 1]},
    )
    case(
        "window",
        "conv_depthwise",
        "Conv",
        [spread(1, 3, 4, 4, seed=26), spread(3, 1, 3, 3, seed=27), spread(3, seed=28)],
        {"group": 3, "pads": [1, 1, 1, 1]},
    )
    case(
        "window",
        "convtranspose_stride2",
        "ConvTranspose",
        [spread(1, 2, 3, 3, seed=29), spread(2, 3, 3, 3, seed=30), spread(3, seed=31)],
        {"strides": [2, 2], "pads": [1, 1, 1, 1], "output_padding": [1, 1]},
    )
    case(
        "window",
        "convtranspose_plain",
        "ConvTranspose",
        [spread(1, 2, 3, 3, seed=29), spread(2, 3, 2, 2, seed=32)],
        {},
    )
    case(
        "window",
        "convtranspose_grouped",
        "ConvTranspose",
        [spread(1, 4, 3, 3, seed=33), spread(4, 2, 3, 3, seed=34), spread(4, seed=35)],
        {"group": 2, "pads": [1, 1, 1, 1]},
    )
    pool_in = spread(1, 2, 5, 5, seed=36)
    case("window", "maxpool_2x2", "MaxPool", [pool_in], {"kernel_shape": [2, 2], "strides": [2, 2]})
    case(
        "window",
        "maxpool_pad",
        "MaxPool",
        [pool_in],
        {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
    )
    case(
        "window",
        "maxpool_ceil",
        "MaxPool",
        [pool_in],
        {"kernel_shape": [2, 2], "strides": [2, 2], "ceil_mode": 1},
    )
    case(
        "window",
        "averagepool_exclude_pad",
        "AveragePool",
        [pool_in],
        {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
    )
    case(
        "window",
        "averagepool_include_pad",
        "AveragePool",
        [pool_in],
        {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1], "count_include_pad": 1},
    )
    case("window", "averagepool_plain", "AveragePool", [pool_in], {"kernel_shape": [2, 2], "strides": [2, 2]})
    case(
        "window",
        "conv_tall_kernel",
        "Conv",
        [image, spread(3, 2, 3, 1, seed=68), bias],
        {"kernel_shape": [3, 1], "pads": [1, 0, 1, 0]},
    )
    case(
        "window",
        "conv_wide_kernel",
        "Conv",
        [image, spread(3, 2, 1, 3, seed=69), bias],
        {"kernel_shape": [1, 3], "pads": [0, 1, 0, 1]},
    )
    case(
        "window",
        "convtranspose_dilated",
        "ConvTranspose",
        [spread(1, 2, 4, 4, seed=70), spread(2, 3, 3, 3, seed=71), spread(3, seed=72)],
        {"dilations": [2, 2], "strides": [2, 1]},
    )
    case("window", "globalaveragepool", "GlobalAveragePool", [pool_in])
    case("window", "globalmaxpool", "GlobalMaxPool", [pool_in])

    # Matrix products.
    case(
        "linear",
        "gemm_transb",
        "Gemm",
        [spread(3, 4, seed=37), spread(5, 4, seed=38), spread(5, seed=39)],
        {"transB": 1, "alpha": 0.7, "beta": 0.3},
    )
    case("linear", "gemm_plain", "Gemm", [spread(3, 4, seed=37), spread(4, 5, seed=40)])
    case(
        "linear",
        "gemm_transa",
        "Gemm",
        [spread(4, 3, seed=41), spread(4, 5, seed=40), spread(3, 5, seed=42)],
        {"transA": 1},
    )
    case("linear", "matmul_2d", "MatMul", [spread(3, 4, seed=37), spread(4, 5, seed=40)])
    case("linear", "matmul_batched", "MatMul", [spread(2, 3, 4, seed=43), spread(2, 4, 5, seed=44)])
    case("linear", "matmul_broadcast", "MatMul", [spread(2, 1, 3, 4, seed=45), spread(4, 5, seed=40)])
    case("linear", "matmul_vector_right", "MatMul", [spread(3, 4, seed=37), spread(4, seed=46)])
    case("linear", "matmul_vector_left", "MatMul", [spread(4, seed=46), spread(4, 5, seed=40)])

    # Remainders. Transformer exports work out head offsets with these.
    mod_a = np.array([7.0, -7.0, 7.0, -7.0, 5.0], dtype=np.float32)
    mod_b = np.array([3.0, 3.0, -3.0, -3.0, 5.0], dtype=np.float32)
    case("elementwise", "mod_fmod", "Mod", [mod_a, mod_b], {"fmod": 1})
    case(
        "elementwise",
        "mod_broadcast",
        "Mod",
        [np.array([[8.0, 9.0], [10.0, 11.0]], dtype=np.float32), np.array([3.0], dtype=np.float32)],
        {"fmod": 1},
    )

    # Recurrence. The weights are [directions, 4H, ...] with the gates ordered
    # i, o, f, c, and the bias holds the input set followed by the recurrent
    # one. `None` stands in for an optional input the case leaves out.
    def lstm(name, seq, batch, inp, hid, direction="forward", bias=True, state=False, **attrs):
        d = 2 if direction == "bidirectional" else 1
        pieces = [
            spread(seq, batch, inp, seed=160),
            spread(d, 4 * hid, inp, low=-0.6, high=0.6, seed=161),
            spread(d, 4 * hid, hid, low=-0.6, high=0.6, seed=162),
        ]
        pieces.append(spread(d, 8 * hid, low=-0.4, high=0.4, seed=163) if bias else None)
        pieces.append(None)
        if state:
            pieces.append(spread(d, batch, hid, low=-0.5, high=0.5, seed=164))
            pieces.append(spread(d, batch, hid, low=-0.5, high=0.5, seed=165))
        case(
            "recurrent", name, "LSTM", pieces,
            {"hidden_size": hid, "direction": direction, **attrs},
            outputs=3, opset=14,
        )

    lstm("lstm_plain", 5, 2, 3, 4, bias=False)
    lstm("lstm_bias", 6, 3, 4, 5)
    lstm("lstm_initial_state", 4, 2, 3, 4, state=True)
    lstm("lstm_reverse", 5, 2, 3, 4, direction="reverse")
    lstm("lstm_bidirectional", 5, 2, 3, 4, direction="bidirectional")
    lstm("lstm_clipped", 5, 2, 3, 4, clip=0.5)
    lstm("lstm_single_step", 1, 1, 2, 3)

    def gru(name, seq, batch, inp, hid, direction="forward", bias=True, state=False, **attrs):
        d = 2 if direction == "bidirectional" else 1
        pieces = [
            spread(seq, batch, inp, seed=170),
            spread(d, 3 * hid, inp, low=-0.6, high=0.6, seed=171),
            spread(d, 3 * hid, hid, low=-0.6, high=0.6, seed=172),
        ]
        pieces.append(spread(d, 6 * hid, low=-0.4, high=0.4, seed=173) if bias else None)
        pieces.append(None)
        if state:
            pieces.append(spread(d, batch, hid, low=-0.5, high=0.5, seed=174))
        case(
            "recurrent", name, "GRU", pieces,
            {"hidden_size": hid, "direction": direction, **attrs},
            outputs=2, opset=14,
        )

    gru("gru_plain", 5, 2, 3, 4, bias=False)
    gru("gru_bias", 6, 3, 4, 5)
    gru("gru_initial_state", 4, 2, 3, 4, state=True)
    gru("gru_reverse", 5, 2, 3, 4, direction="reverse")
    gru("gru_bidirectional", 5, 2, 3, 4, direction="bidirectional")
    gru("gru_clipped", 5, 2, 3, 4, clip=0.5)
    gru("gru_linear_before_reset", 5, 2, 3, 4, linear_before_reset=1)
    gru("gru_linear_bidirectional", 4, 2, 3, 4, direction="bidirectional", linear_before_reset=1)
    gru("gru_single_step", 1, 1, 2, 3)

    # Normalisation.
    case(
        "norm",
        "batchnorm",
        "BatchNormalization",
        [
            spread(2, 3, 2, 2, seed=47),
            spread(3, seed=48),
            spread(3, seed=49),
            spread(3, seed=50),
            np.abs(spread(3, seed=51)) + 0.5,
        ],
        {"epsilon": 1e-3},
    )
    case(
        "norm",
        "batchnorm_2d",
        "BatchNormalization",
        [
            spread(4, 3, seed=52),
            spread(3, seed=48),
            spread(3, seed=49),
            spread(3, seed=50),
            np.abs(spread(3, seed=51)) + 0.5,
        ],
        {},
    )
    case(
        "norm",
        "instancenorm",
        "InstanceNormalization",
        [spread(2, 3, 2, 3, seed=53), spread(3, seed=54), spread(3, seed=55)],
        {"epsilon": 1e-4},
    )
    case(
        "norm",
        "lrn",
        "LRN",
        [spread(1, 5, 2, 2, seed=56)],
        {"size": 3, "alpha": 0.0002, "beta": 0.6, "bias": 1.5},
    )
    case("norm", "lrn_defaults", "LRN", [spread(2, 4, 2, 2, seed=57)], {"size": 5})
    # Layer normalisation, which every transformer export opens with. The
    # scale and shift take the shape of the axes being normalised, not their
    # flattened length, which is what the operator's own checker enforces.
    case(
        "norm",
        "layernorm_last",
        "LayerNormalization",
        [spread(4, 16, seed=140), spread(16, seed=141), spread(16, seed=142)],
        {"epsilon": 1e-5},
    )
    case(
        "norm",
        "layernorm_tokens",
        "LayerNormalization",
        [spread(2, 7, 32, seed=143), spread(32, seed=144), spread(32, seed=145)],
        {"epsilon": 1e-5},
    )
    case(
        "norm",
        "layernorm_axis_one",
        "LayerNormalization",
        [spread(2, 3, 4, seed=146), spread(3, 4, seed=147), spread(3, 4, seed=148)],
        {"axis": 1, "epsilon": 1e-3},
    )
    case(
        "norm",
        "layernorm_no_bias",
        "LayerNormalization",
        [spread(3, 8, seed=149), spread(8, seed=150)],
        {"epsilon": 1e-6},
    )

    # Reductions.
    reduce_in = spread(2, 3, 4, seed=58)
    for op in ["ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin"]:
        low = op.lower()
        if op == "ReduceSum":
            case("reduce", f"{low}_axis1", op, [reduce_in, np.array([1], dtype=np.int64)])
            case(
                "reduce",
                f"{low}_axes02",
                op,
                [reduce_in, np.array([0, 2], dtype=np.int64)],
                {"keepdims": 0},
            )
            case("reduce", f"{low}_all", op, [reduce_in], {"keepdims": 0})
        else:
            case("reduce", f"{low}_axis1", op, [reduce_in], {"axes": [1]}, opset=12)
            case("reduce", f"{low}_axes02", op, [reduce_in], {"axes": [0, 2], "keepdims": 0}, opset=12)
            case("reduce", f"{low}_all", op, [reduce_in], {"keepdims": 0}, opset=12)
    ties = np.array([[1.0, 3.0, 3.0, 2.0], [5.0, -1.0, 5.0, 0.0]], dtype=np.float32)
    case("reduce", "argmax_axis1", "ArgMax", [ties], {"axis": 1, "keepdims": 0})
    case("reduce", "argmax_last_index", "ArgMax", [ties], {"axis": 1, "keepdims": 0, "select_last_index": 1})
    case("reduce", "argmin_axis0", "ArgMin", [ties], {"axis": 0})
    case("reduce", "argmax_axis0_keepdims", "ArgMax", [reduce_in], {"axis": 0})
    case("reduce", "reducemean_negative_axis", "ReduceMean", [reduce_in], {"axes": [-1]}, opset=12)

    # Resampling.
    picture = spread(1, 2, 3, 4, seed=59)
    for transform in ["half_pixel", "asymmetric", "align_corners", "pytorch_half_pixel"]:
        case(
            "resample",
            f"resize_linear_{transform}",
            "Resize",
            [picture, None, None, np.array([1, 2, 6, 7], dtype=np.int64)],
            {"mode": "linear", "coordinate_transformation_mode": transform},
        )
    case(
        "resample",
        "resize_nearest_sizes",
        "Resize",
        [picture, None, None, np.array([1, 2, 6, 7], dtype=np.int64)],
        {"mode": "nearest", "coordinate_transformation_mode": "asymmetric"},
    )
    case(
        "resample",
        "resize_nearest_scales",
        "Resize",
        [picture, None, np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)],
        {"mode": "nearest", "coordinate_transformation_mode": "asymmetric"},
    )
    case(
        "resample",
        "resize_down_linear",
        "Resize",
        [spread(1, 1, 6, 6, seed=60), None, None, np.array([1, 1, 3, 3], dtype=np.int64)],
        {"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
    )
    case(
        "resample",
        "upsample_nearest",
        "Upsample",
        [picture, np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)],
        {"mode": "nearest"},
        opset=9,
    )
    case(
        "resample",
        "upsample_linear",
        "Upsample",
        [picture, np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)],
        {"mode": "linear"},
        opset=9,
    )


# ------------------------------------------------------------- running them


def emit_ops(path):
    import onnx
    import onnxruntime as ort
    from onnx import helper

    build_cases()
    lines = [
        "# One node a case, run through onnxruntime.",
        "# Rebuild with packages/jaicv/tools/dnn_graph_oracle.py -- do not edit by hand.",
    ]
    group = None
    for spec in CASES:
        names = []
        info = []
        feed = {}
        for index, array in enumerate(spec["inputs"]):
            if array is None:
                names.append("")
                continue
            array = np.asarray(array)
            name = f"in{index}"
            names.append(name)
            info.append(
                helper.make_tensor_value_info(
                    name, onnx.helper.np_dtype_to_tensor_dtype(array.dtype), list(array.shape)
                )
            )
            feed[name] = array
        outs = [f"out{slot}" for slot in range(spec["outputs"])]
        attrs = {
            key: (
                onnx.numpy_helper.from_array(value.array, name=key)
                if isinstance(value, TensorAttr)
                else value
            )
            for key, value in spec["attrs"].items()
        }
        node = helper.make_node(spec["op"], names, outs, **attrs)
        graph = helper.make_graph(
            [node], spec["name"], info, [helper.make_empty_tensor_value_info(o) for o in outs]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", spec["opset"])])
        model.ir_version = 9
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        produced = session.run(None, feed)

        if spec["group"] != group:
            group = spec["group"]
            lines.append(f"group {group}")
        lines.append(f"case {spec['op']} {spec['name']}")
        for key, value in spec["attrs"].items():
            lines.append(attr_line(key, value))
        for array in spec["inputs"]:
            lines.append("in null" if array is None else tensor_line("in", array))
        for array in produced:
            lines.append(tensor_line("out", array))
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {len(CASES)} cases to {path}")


def emit_e2e(path):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(7)
    x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
    weight = torch.randn(4, 3, 3, 3, dtype=torch.float64) * 0.4
    bias = torch.randn(4, dtype=torch.float64) * 0.2
    gamma = torch.rand(4, dtype=torch.float64) + 0.5
    beta = torch.randn(4, dtype=torch.float64) * 0.3
    mean = torch.randn(4, dtype=torch.float64) * 0.2
    var = torch.rand(4, dtype=torch.float64) + 0.4
    dense = torch.randn(5, 4 * 4 * 4, dtype=torch.float64) * 0.1
    dense_bias = torch.randn(5, dtype=torch.float64) * 0.1

    y = F.conv2d(x, weight, bias, stride=1, padding=1)
    y = F.batch_norm(y, mean, var, gamma, beta, training=False, eps=1e-5)
    y = F.relu(y)
    y = F.max_pool2d(y, 2, 2)
    y = y.reshape(y.shape[0], -1)
    y = F.linear(y, dense, dense_bias)
    y = F.softmax(y, dim=1)

    lines = [
        "# conv, batch norm, relu, max pool, flatten, gemm, softmax -- run in torch.",
        "# Rebuild with packages/jaicv/tools/dnn_graph_oracle.py -- do not edit by hand.",
        "group network",
        "case Network conv_bn_relu_pool_flatten_gemm_softmax",
    ]
    for tensor in [x, weight, bias, gamma, beta, mean, var, dense, dense_bias]:
        lines.append(tensor_line("in", tensor.detach().numpy()))
    lines.append(tensor_line("out", y.detach().numpy()))
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote the network oracle to {path}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("ops", "e2e"):
        raise SystemExit("usage: dnn_graph_oracle.py ops|e2e <output path>")
    if sys.argv[1] == "ops":
        emit_ops(sys.argv[2])
    else:
        emit_e2e(sys.argv[2])
