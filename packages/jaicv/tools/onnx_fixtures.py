#!/usr/bin/env python3
"""Write the ONNX fixtures in `tests/models/` and record what they produce.

Five of the models are real torch exports -- an MLP, a CNN, a residual net, a
net that concatenates two branches, and one that reshapes and transposes --
run in torch afterwards so `tests/test_onnx.jai` has torch's own answer to
compare against. The rest are built with `onnx.helper` because they hold
things torch will not export: every tensor data type the importer decodes, an
opset-11 Softmax over a rank-4 input (whose reference comes from
onnxruntime, since torch cannot express the pre-13 semantics), a node the
executor has no operator for, and an initializer whose data lives in a file
that is not there.

    cd /tmp && uv run --with onnx --with onnxruntime --with numpy --with torch \\
        python ~/Developer/projects/jaithon/packages/jaicv/tools/onnx_fixtures.py

Everything it writes is checked in, so the test needs none of those packages.
"""
from __future__ import annotations

import pathlib
import struct

import numpy as np
import onnx
import torch
from torch import nn
from onnx import TensorProto, helper

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODELS = ROOT / "packages" / "jaicv" / "tests" / "models"
CASES = MODELS / "onnx_cases.txt"

OPSET = 17

lines: list[str] = []


def number(value: float) -> str:
    """One value, in a spelling Jaithon's `float()` reads back exactly."""
    value = float(value)
    if value != value:
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return repr(value)


def record(kind: str, name: str, array: np.ndarray) -> None:
    array = np.asarray(array)
    shape = list(array.shape)
    values = [number(v) for v in array.reshape(-1).tolist()]
    lines.append(
        " ".join([kind, name, str(len(shape))] + [str(d) for d in shape] + values)
    )


# ------------------------------------------------------------ torch exports


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.one = nn.Linear(6, 8)
        self.two = nn.Linear(8, 4)

    def forward(self, x):
        return self.two(torch.relu(self.one(x)))


class Cnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2)
        self.dense = nn.Linear(4 * 4 * 4, 5)

    def forward(self, x):
        x = self.pool(torch.relu(self.norm(self.conv(x))))
        return torch.softmax(self.dense(x.flatten(1)), dim=1)


class Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.one = nn.Conv2d(2, 2, 3, padding=1)
        self.two = nn.Conv2d(2, 2, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.two(torch.relu(self.one(x))))


class Concat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(5, 3)
        self.right = nn.Linear(5, 4)
        self.head = nn.Linear(7, 2)

    def forward(self, x):
        joined = torch.cat([torch.relu(self.left(x)), torch.tanh(self.right(x))], dim=1)
        return self.head(joined)


class Shuffle(nn.Module):
    """A transpose and a reshape that a shape inference cannot fold away."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(12, 3)

    def forward(self, x):
        moved = x.permute(0, 2, 1)
        return self.head(moved.reshape(x.shape[0], 12))


def export(
    model: nn.Module,
    sample: torch.Tensor,
    name: str,
    fold: bool,
    run_on: torch.Tensor | None = None,
    **extra,
) -> None:
    model.eval()
    path = MODELS / name
    torch.onnx.export(
        model,
        (sample,),
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET,
        do_constant_folding=fold,
        dynamo=False,
        **extra,
    )
    #: A model exported with a symbolic batch is recorded at a different batch
    #: from the one it was traced at, so that the dimension is actually free.
    sample = sample if run_on is None else run_on
    with torch.no_grad():
        produced = model(sample)
    ops = sorted({node.op_type for node in onnx.load(path).graph.node})
    print(f"{name:24s} {path.stat().st_size:6d} bytes  {' '.join(ops)}")
    lines.append(f"model {name}")
    record("feed", "input", sample.numpy())
    record("want", "output", produced.numpy())


def torch_fixtures() -> None:
    torch.manual_seed(7)
    export(Mlp(), torch.randn(2, 6), "onnx_mlp.onnx", True)
    #: Folding is off so the batch norm survives as its own node rather than
    #: being multiplied into the convolution's weights.
    export(Cnn(), torch.randn(1, 1, 8, 8), "onnx_cnn.onnx", False)
    export(Residual(), torch.randn(1, 2, 4, 4), "onnx_residual.onnx", True)
    export(Concat(), torch.randn(3, 5), "onnx_concat.onnx", True)
    export(Shuffle(), torch.randn(2, 4, 3), "onnx_shuffle.onnx", True)
    #: The old exporter default, and what a Caffe2-era file looks like: every
    #: weight is listed as a graph input beside the image.
    export(
        Mlp(),
        torch.randn(2, 6),
        "onnx_weights_as_inputs.onnx",
        True,
        keep_initializers_as_inputs=True,
    )
    export(
        Mlp(),
        torch.randn(2, 6),
        "onnx_dynamic_batch.onnx",
        True,
        run_on=torch.randn(5, 6),
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )


# --------------------------------------------------------- data type fixture

#: Each entry is a name, the ONNX data type, the values, and whether they go
#: into `raw_data` or into the repeated field the type is stored in. Both
#: routes are covered for every width, because an exporter picks either.
DTYPE_CASES = [
    ("float_raw", TensorProto.FLOAT, [0.0, -1.5, 3.25e30, 1.401298464324817e-45], True),
    ("float_typed", TensorProto.FLOAT, [1.0, -0.0, float("inf"), float("nan")], False),
    ("uint8_raw", TensorProto.UINT8, [0, 1, 128, 255], True),
    ("uint8_typed", TensorProto.UINT8, [0, 255], False),
    ("int8_raw", TensorProto.INT8, [-128, -1, 0, 127], True),
    ("int8_typed", TensorProto.INT8, [-128, 127], False),
    ("uint16_raw", TensorProto.UINT16, [0, 1, 32768, 65535], True),
    ("uint16_typed", TensorProto.UINT16, [0, 65535], False),
    ("int16_raw", TensorProto.INT16, [-32768, -1, 0, 32767], True),
    ("int16_typed", TensorProto.INT16, [-32768, 32767], False),
    ("int32_raw", TensorProto.INT32, [-2147483648, -1, 0, 2147483647], True),
    ("int32_typed", TensorProto.INT32, [-2147483648, 2147483647], False),
    ("int64_raw", TensorProto.INT64, [-9223372036854775808, -1, 0, 9223372036854775807], True),
    ("int64_typed", TensorProto.INT64, [-4294967296, 4294967296], False),
    ("bool_raw", TensorProto.BOOL, [True, False, True, True], True),
    ("bool_typed", TensorProto.BOOL, [False, True], False),
    ("float16_raw", TensorProto.FLOAT16, [0.0, -2.0, 65504.0, 5.960464477539063e-08], True),
    ("float16_typed", TensorProto.FLOAT16, [1.0, 6.103515625e-05, float("inf"), float("nan")], False),
    ("double_raw", TensorProto.DOUBLE, [0.0, -1.0, 1.7976931348623157e308, 2.5e-320], True),
    ("double_typed", TensorProto.DOUBLE, [0.1, -1e300], False),
    ("uint32_raw", TensorProto.UINT32, [0, 1, 2147483648, 4294967295], True),
    ("uint32_typed", TensorProto.UINT32, [0, 4294967295], False),
    ("uint64_raw", TensorProto.UINT64, [0, 1, 9223372036854775808, 18446744073709551615], True),
    ("uint64_typed", TensorProto.UINT64, [0, 18446744073709551615], False),
    ("bfloat16_raw", TensorProto.BFLOAT16, [0.0, -2.5, 3.3895313892515355e38, 9.183549615799121e-41], True),
    ("bfloat16_typed", TensorProto.BFLOAT16, [1.0, -1.0], False),
]

NUMPY_OF = {
    TensorProto.FLOAT: np.float32,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT8: np.int8,
    TensorProto.UINT16: np.uint16,
    TensorProto.INT16: np.int16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.BOOL: np.bool_,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
    TensorProto.UINT32: np.uint32,
    TensorProto.UINT64: np.uint64,
}


def bfloat16_bits(value: float) -> int:
    """The top sixteen bits of the float32 nearest `value`, rounding to even."""
    bits = struct.unpack("<I", struct.pack("<f", np.float32(value)))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded & 0xFFFF


def bfloat16_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def dtype_tensor(name: str, kind: int, values: list, raw: bool) -> tuple:
    """A `TensorProto` and the float32 values reading it back should give."""
    if kind == TensorProto.BFLOAT16:
        bits = [bfloat16_bits(v) for v in values]
        exact = np.array([bfloat16_value(b) for b in bits], dtype=np.float32)
        if raw:
            tensor = helper.make_tensor(
                name, kind, [len(values)], b"".join(struct.pack("<H", b) for b in bits), raw=True
            )
        else:
            #: Unlike the raw route, `make_tensor` takes bfloat16 and float16
            #: as the values themselves and rounds them into `int32_data` on
            #: the way -- handing it the bit patterns stores the bit patterns
            #: as numbers.
            tensor = helper.make_tensor(name, kind, [len(values)], values, raw=False)
        return tensor, exact
    array = np.array(values, dtype=NUMPY_OF[kind])
    exact = array.astype(np.float32)
    if raw:
        tensor = helper.make_tensor(name, kind, [len(values)], array.tobytes(), raw=True)
    else:
        tensor = helper.make_tensor(name, kind, [len(values)], array.reshape(-1).tolist(), raw=False)
    return tensor, exact


def dtype_fixture() -> None:
    """Every data type as an initializer, beside a graph that does nothing.

    The initializers are read straight out of the imported graph rather than
    run: what is being checked is the decoding, and putting each one through
    an operator would only add the operator's arithmetic to the comparison.
    """
    initializers = []
    lines.append("model onnx_dtypes.onnx")
    for (name, kind, values, raw) in DTYPE_CASES:
        tensor, exact = dtype_tensor(name, kind, values, raw)
        initializers.append(tensor)
        record("init", name, exact)
    #: A rank-0 initializer and one that is stored in `raw_data` with dims the
    #: importer has to trust rather than infer.
    scalar = helper.make_tensor("scalar", TensorProto.FLOAT, [], [2.5])
    initializers.append(scalar)
    record("init", "scalar", np.array([2.5], dtype=np.float32))
    matrix = helper.make_tensor(
        "matrix", TensorProto.FLOAT, [2, 3], np.arange(6, dtype=np.float32).tobytes(), raw=True
    )
    initializers.append(matrix)
    record("init", "matrix", np.arange(6, dtype=np.float32).reshape(2, 3))

    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"], name="passthrough")],
        "dtypes",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    onnx.save(model, MODELS / "onnx_dtypes.onnx")
    print(f"onnx_dtypes.onnx          {(MODELS / 'onnx_dtypes.onnx').stat().st_size} bytes")


# ------------------------------------------------------- hand-built fixtures


def legacy_softmax_fixture(name: str, **attrs) -> None:
    """Opset 11 Softmax over a rank-4 input, where the pre-13 rule differs.

    Before opset 13 a Softmax flattened its input to two dimensions at `axis`
    and normalised whole rows of that; from 13 it normalises along `axis`
    alone. On a rank-4 input the two disagree, so this is the case that says
    whether the importer rewrote the node or merely copied it. It is written
    twice, once with the axis left out -- where the pre-13 default of 1 is
    also not the modern default of -1 -- and once with an axis of 2.
    """
    graph = helper.make_graph(
        [helper.make_node("Softmax", ["input"], ["output"], name="legacy", **attrs)],
        "legacy_softmax",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 2, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    path = MODELS / name
    onnx.save(model, path)

    rng = np.random.default_rng(11)
    sample = rng.standard_normal((2, 3, 2, 2)).astype(np.float32)
    lines.append(f"model {name}")
    record("feed", "input", sample)
    record("want", "output", onnxruntime_run(path, sample))
    print(f"{name:26s}{path.stat().st_size} bytes  (onnxruntime reference)")

    #: The same model with its opset declaration removed, which leaves nothing
    #: to say which of the two Softmax rules the file meant.
    if "axis" in attrs:
        model.ClearField("opset_import")
        onnx.save(model, MODELS / "onnx_no_opset.onnx")


def short_weight_fixture() -> None:
    """An initializer whose `raw_data` is a byte short of its dimensions."""
    weight = helper.make_tensor(
        "weight", TensorProto.FLOAT, [4], np.arange(4, dtype=np.float32).tobytes(), raw=True
    )
    weight.raw_data = weight.raw_data[:-1]
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", "weight"], ["output"], name="biased")],
        "short",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    onnx.save(model, MODELS / "onnx_short_weight.onnx")


def unsupported_fixture() -> None:
    """A graph holding `Celu`, which is a real ONNX operator the executor lacks."""
    graph = helper.make_graph(
        [helper.make_node("Celu", ["input"], ["output"], name="curved", alpha=1.0)],
        "unsupported",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    onnx.save(model, MODELS / "onnx_unsupported.onnx")


def custom_domain_fixture() -> None:
    """A node from somebody else's operator set."""
    node = helper.make_node(
        "FusedConv", ["input"], ["output"], name="fused", domain="com.microsoft"
    )
    graph = helper.make_graph(
        [node],
        "custom",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET), helper.make_opsetid("com.microsoft", 1)],
    )
    onnx.save(model, MODELS / "onnx_custom_domain.onnx")


def external_fixture() -> None:
    """An initializer whose bytes are in a file nobody shipped."""
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [4], [0.0, 0.0, 0.0, 0.0])
    weight.ClearField("float_data")
    weight.data_location = TensorProto.EXTERNAL
    entry = weight.external_data.add()
    entry.key = "location"
    entry.value = "weight.bin"
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", "weight"], ["output"], name="biased")],
        "external",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    onnx.save(model, MODELS / "onnx_external.onnx")


def constant_fixture() -> None:
    """Constant nodes in each of the five shapes an exporter writes them in.

    The two integer constants hang off the side of the graph rather than
    taking part in its arithmetic: ONNX will not add an int64 to a float, and
    what is being checked about them is that the importer decoded them at all,
    which the test reads off the imported graph directly.
    """
    tensor = helper.make_tensor("held", TensorProto.FLOAT, [5], [1.0, 2.0, 3.0, 4.0, 5.0])
    nodes = [
        helper.make_node("Constant", [], ["from_tensor"], name="c_tensor", value=tensor),
        helper.make_node("Constant", [], ["from_float"], name="c_float", value_float=0.5),
        helper.make_node(
            "Constant",
            [],
            ["from_floats"],
            name="c_floats",
            value_floats=[1.5, -2.5, 0.25, -0.75, 2.0],
        ),
        helper.make_node("Constant", [], ["from_int"], name="c_int", value_int=7),
        helper.make_node("Constant", [], ["from_ints"], name="c_ints", value_ints=[4, 5, 6]),
        helper.make_node("Add", ["input", "from_tensor"], ["sum"], name="add_held"),
        helper.make_node("Mul", ["sum", "from_float"], ["scaled"], name="halve"),
        helper.make_node("Sub", ["scaled", "from_floats"], ["output"], name="finish"),
    ]
    graph = helper.make_graph(
        nodes,
        "constants",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [5])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [5])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    path = MODELS / "onnx_constants.onnx"
    onnx.save(model, path)

    sample = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    lines.append("model onnx_constants.onnx")
    record("feed", "input", sample)
    record("want", "output", onnxruntime_run(path, sample))
    record("init", "from_int", np.array([7.0], dtype=np.float32))
    record("init", "from_ints", np.array([4.0, 5.0, 6.0], dtype=np.float32))
    print(f"onnx_constants.onnx       {path.stat().st_size} bytes")


def onnxruntime_run(path: pathlib.Path, sample: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(["output"], {"input": sample})[0]


def untyped_attributes_fixture() -> None:
    """The same attributes with `AttributeProto.type` cleared.

    An exporter old enough to predate that field leaves the shape of an
    attribute to be worked out from which payload field is present. The model
    is run through onnxruntime before the field is stripped, because stripping
    it is exactly the thing being tested and onnxruntime will not read the
    result.
    """
    pads = helper.make_tensor("pads", TensorProto.INT64, [4], [0, 0, 0, 1])
    bias = [0.5, -0.25, 0.125, 1.0, -1.0, 2.0, 0.0, 0.75, -0.5, 0.25, -0.125, 1.5]
    nodes = [
        helper.make_node("Transpose", ["input"], ["moved"], name="move", perm=[0, 2, 1]),
        helper.make_node("Flatten", ["moved"], ["flat"], name="flatten", axis=1),
        helper.make_node("LeakyRelu", ["flat"], ["leaky"], name="leak", alpha=0.125),
        helper.make_node("Constant", [], ["bias"], name="c_bias", value_floats=bias),
        helper.make_node("Add", ["leaky", "bias"], ["biased"], name="bias_it"),
        helper.make_node("Pad", ["biased", "pads"], ["output"], name="edge", mode="edge"),
    ]
    graph = helper.make_graph(
        nodes,
        "untyped",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 13])],
        initializer=[pads],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    path = MODELS / "onnx_untyped_attrs.onnx"
    onnx.save(model, path)

    rng = np.random.default_rng(3)
    sample = rng.standard_normal((2, 3, 4)).astype(np.float32)
    produced = onnxruntime_run(path, sample)

    for node in model.graph.node:
        for attribute in node.attribute:
            attribute.ClearField("type")
    onnx.save(model, path)
    lines.append("model onnx_untyped_attrs.onnx")
    record("feed", "input", sample)
    record("want", "output", produced)
    record("init", "bias", np.array(bias, dtype=np.float32))
    print(f"onnx_untyped_attrs.onnx   {path.stat().st_size} bytes")


def legacy_reshape_fixture() -> None:
    """Reshape as opset 1 wrote it, with the shape as an attribute."""
    graph = helper.make_graph(
        [helper.make_node("Reshape", ["input"], ["output"], name="fold", shape=[4, 6])],
        "legacy_reshape",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 6])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 4)])
    model.ir_version = 3
    path = MODELS / "onnx_legacy_reshape.onnx"
    onnx.save(model, path)

    rng = np.random.default_rng(4)
    sample = rng.standard_normal((2, 3, 4)).astype(np.float32)
    lines.append("model onnx_legacy_reshape.onnx")
    record("feed", "input", sample)
    record("want", "output", onnxruntime_run(path, sample))
    print(f"onnx_legacy_reshape.onnx  {path.stat().st_size} bytes")


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    torch_fixtures()
    dtype_fixture()
    legacy_softmax_fixture("onnx_legacy_softmax.onnx")
    legacy_softmax_fixture("onnx_legacy_softmax_axis.onnx", axis=2)
    short_weight_fixture()
    unsupported_fixture()
    custom_domain_fixture()
    external_fixture()
    constant_fixture()
    untyped_attributes_fixture()
    legacy_reshape_fixture()
    CASES.write_text(
        "# Written by packages/jaicv/tools/onnx_fixtures.py -- do not edit by hand.\n"
        + "\n".join(lines)
        + "\n"
    )
    print(f"{CASES.name}: {len(lines)} lines")


if __name__ == "__main__":
    main()
