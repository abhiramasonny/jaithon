#: Reading an ONNX model.
#:
#: An `.onnx` file is one `ModelProto` in protobuf's wire format, so the
#: reading of it belongs to `protobuf.jai` and what is left here is the schema
#: -- which field number means what -- and the translation into the `Graph`
#: the shared executor runs. That translation is unusually thin, because the
#: executor already speaks ONNX: an operator name goes straight across. Nearly
#: all of this module is the other two jobs, which are getting the weights out
#: of whichever encoding they arrived in and papering over the places where an
#: operator's meaning shifted under it between opsets.
#:
#: Every field number below was read off onnx 1.22's own descriptors rather
#: than a document; `tools/onnx_fixtures.py` prints them again.
#:
#: Three things here are load-bearing.
#:
#: A tensor's numbers arrive in one of two encodings, and the same type does
#: not use the same bytes in both. `raw_data` is fixed-width little-endian, so
#: an `int32` there is always four bytes, while `int32_data` is a packed varint
#: repeat, so the same number is one to ten. A reader that hands `raw_data` to
#: a varint decoder gets numbers rather than an error, which is the worst way
#: to be wrong, so the two routes are separate below and the raw one checks its
#: length against the dimensions before it reads a byte.
#:
#: Before opset 13 a `Softmax` did not normalise along its axis. It flattened
#: the input to two dimensions at that axis and normalised whole rows of the
#: result, which above rank two is a different answer, and its default axis was
#: 1 rather than -1. The executor implements the modern rule only, so a pre-13
#: `Softmax` is rewritten here into the four operators that spell the old rule
#: out -- `Shape`, `Flatten`, `Softmax`, `Reshape` -- rather than passed
#: through to quietly mean something else.
#:
#: `Constant` nodes are folded into initializers as the file is read. Partly
#: that saves the executor a node, but mostly it collects the weights: what a
#: file holds as a mix of initializers and constants comes back as one
#: dictionary, and a model is freed by walking it.
#:
#: Ownership: every tensor an imported graph holds is this module's doing, and
#: `free_onnx_weights` releases all of them -- the initializers and the tensor
#: an attribute may carry. Nothing else in the graph touches the GPU.
#:
#: ```
#: let graph = read_net_from_onnx("mobilenet.onnx")
#: defer { free_onnx_weights(graph) }
#: let outputs = run_graph(graph, {"input": blob})
#: ```
import std.io as io
import std.math
from jaitensor import Tensor
from ..core.mat import Mat
from ..core.types import CV_32F, CV_MAKETYPE
from .blob import Net, Runnable
from .graph import Attr, Graph, GraphNode, is_implemented, run_graph
from .plan import run_best
from .protobuf import Field, WIRE_I32, WIRE_I64, WIRE_LEN, WIRE_VARINT, as_bytes, as_double, as_float, as_string, field_by_number, fields_by_number, packed_doubles, packed_fixed32, packed_fixed64, packed_floats, packed_varints, parse_message, parse_nested

#: ModelProto.
const MODEL_GRAPH = 7
const MODEL_OPSET = 8

#: OperatorSetIdProto.
const OPSET_DOMAIN = 1
const OPSET_VERSION = 2

#: GraphProto.
const GRAPH_NODE = 1
const GRAPH_INITIALIZER = 5
const GRAPH_INPUT = 11
const GRAPH_OUTPUT = 12
const GRAPH_SPARSE_INITIALIZER = 15

#: NodeProto.
const NODE_INPUT = 1
const NODE_OUTPUT = 2
const NODE_NAME = 3
const NODE_OP_TYPE = 4
const NODE_ATTRIBUTE = 5
const NODE_DOMAIN = 7

#: AttributeProto.
const ATTR_NAME = 1
const ATTR_FLOAT = 2
const ATTR_INT = 3
const ATTR_STRING = 4
const ATTR_TENSOR = 5
const ATTR_GRAPH = 6
const ATTR_FLOATS = 7
const ATTR_INTS = 8
const ATTR_STRINGS = 9
const ATTR_TENSORS = 10
const ATTR_GRAPHS = 11
const ATTR_KIND = 20
const ATTR_REF = 21
const ATTR_SPARSE = 22
const ATTR_SPARSES = 23

#: AttributeProto.AttributeType, for the shapes an attribute can take. The
#: seven the executor's `Attr` has a field for are named; the rest are refused
#: by number.
const KIND_UNDEFINED = 0
const KIND_FLOAT = 1
const KIND_INT = 2
const KIND_STRING = 3
const KIND_TENSOR = 4
const KIND_GRAPH = 5
const KIND_FLOATS = 6
const KIND_INTS = 7
const KIND_STRINGS = 8

#: TensorProto.
const TENSOR_DIMS = 1
const TENSOR_DATA_TYPE = 2
const TENSOR_SEGMENT = 3
const TENSOR_FLOAT_DATA = 4
const TENSOR_INT32_DATA = 5
const TENSOR_INT64_DATA = 7
const TENSOR_NAME = 8
const TENSOR_RAW_DATA = 9
const TENSOR_DOUBLE_DATA = 10
const TENSOR_UINT64_DATA = 11
const TENSOR_EXTERNAL = 13
const TENSOR_LOCATION = 14

#: StringStringEntryProto, which is how one entry of `external_data` is spelt.
const ENTRY_KEY = 1
const ENTRY_VALUE = 2

#: ValueInfoProto and the TypeProto chain hanging under it, which is where a
#: declared input shape lives.
const VALUE_NAME = 1
const VALUE_TYPE = 2
const TYPE_TENSOR = 1
const ELEM_SHAPE = 2
const SHAPE_DIM = 1
const DIM_VALUE = 1

#: TensorProto.DataType.
const DT_FLOAT = 1
const DT_UINT8 = 2
const DT_INT8 = 3
const DT_UINT16 = 4
const DT_INT16 = 5
const DT_INT32 = 6
const DT_INT64 = 7
const DT_STRING = 8
const DT_BOOL = 9
const DT_FLOAT16 = 10
const DT_DOUBLE = 11
const DT_UINT32 = 12
const DT_UINT64 = 13
const DT_BFLOAT16 = 16

#: TensorProto.DataLocation.
const LOCATION_EXTERNAL = 1

#: The opset at which `Softmax` and `LogSoftmax` stopped coercing to two
#: dimensions and started normalising along one axis.
const AXIS_SOFTMAX_OPSET = 13

#: `2**64`, which turns the signed reading of a `uint64` back into its
#: magnitude. Protobuf has no unsigned integer wider than `int` holds, so a
#: `uint64` above `2**63 - 1` reaches this module as the negative sharing its
#: bit pattern.
const TWO_TO_64 = 18446744073709551616.0

#: What each data type is called when an error has to name one.
let DATA_TYPE_NAMES: dict[int, str] = {
    0: "UNDEFINED",
    DT_FLOAT: "FLOAT",
    DT_UINT8: "UINT8",
    DT_INT8: "INT8",
    DT_UINT16: "UINT16",
    DT_INT16: "INT16",
    DT_INT32: "INT32",
    DT_INT64: "INT64",
    DT_STRING: "STRING",
    DT_BOOL: "BOOL",
    DT_FLOAT16: "FLOAT16",
    DT_DOUBLE: "DOUBLE",
    DT_UINT32: "UINT32",
    DT_UINT64: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
    DT_BFLOAT16: "BFLOAT16",
}

#: Bytes one element takes in `raw_data`.
#:
#: This doubles as the list of types this reader decodes: a type absent from
#: it is refused by both routes, whichever one it arrived by.
let RAW_WIDTHS: dict[int, int] = {
    DT_FLOAT: 4,
    DT_UINT8: 1,
    DT_INT8: 1,
    DT_UINT16: 2,
    DT_INT16: 2,
    DT_INT32: 4,
    DT_INT64: 8,
    DT_BOOL: 1,
    DT_FLOAT16: 2,
    DT_DOUBLE: 8,
    DT_UINT32: 4,
    DT_UINT64: 8,
    DT_BFLOAT16: 2,
}

fn data_type_name(kind: int) -> str {
    return DATA_TYPE_NAMES.get(kind, f"the data type numbered {kind}")
}

fn le_u16(data: bytes, at: int) -> int { return data[at] | data[at + 1] << 8 }

#: An IEEE-754 binary16 out of its sixteen bits.
#:
#: Assembled with `ldexp` rather than by scaling the fraction, because a biased
#: exponent of zero means the leading one is *absent*: a subnormal is the
#: fraction times `2**-24` outright, and adding the hidden bit first puts every
#: one of them out by `2**-14`.
fn f16_from_bits(bits: int) -> float {
    let exponent = bits >> 10 & 0x1F
    let fraction = bits & 0x3FF
    var value = 0.0
    if exponent == 0 {
        value = math.ldexp(float(fraction), -24)
    } elif exponent == 0x1F {
        value = if fraction == 0 { math.INF } else { math.NAN }
    } else {
        value = math.ldexp(float(fraction + 1024), exponent - 25)
    }
    return if bits >> 15 & 1 == 1 { -value } else { value }
}

#: An IEEE-754 bfloat16 out of its sixteen bits.
#:
#: A bfloat16 is a binary32 with its low sixteen bits cut off and nothing else,
#: so putting them back is the whole conversion. This is the one float in a
#: model file that `protobuf.jai`'s readers cannot decode on the way past --
#: every other arrives as a whole `fixed32` or `fixed64`.
fn bf16_from_bits(bits: int) -> float {
    let whole = bits << 16
    let exponent = whole >> 23 & 0xFF
    let fraction = whole & 0x7FFFFF
    var value = 0.0
    if exponent == 0 {
        value = math.ldexp(float(fraction), -149)
    } elif exponent == 0xFF {
        value = if fraction == 0 { math.INF } else { math.NAN }
    } else {
        value = math.ldexp(float(fraction + 0x800000), exponent - 150)
    }
    return if whole >> 31 & 1 == 1 { -value } else { value }
}

#: A `uint64` as its magnitude rather than as the negative it shares a bit
#: pattern with.
fn unsigned_64(value: int) -> float {
    return if value < 0 { float(value) + TWO_TO_64 } else { float(value) }
}

#: A `fixed32` read as the `int32` it encodes.
fn signed_32(value: int) -> int {
    return if value >= 0x80000000 { value - 0x100000000 } else { value }
}

#: Every value of a repeated varint field, whether the writer packed it or not.
#:
#: Which it did is not a matter of taste: ONNX marks `TensorProto`'s data
#: fields packed and leaves `AttributeProto`'s `ints` unpacked, so both forms
#: turn up in one file, and a merge can split either across several runs.
fn repeated_varints(fields: list[Field], number: int, where: str) -> list[int] {
    var out: list[int] = []
    for field in fields_by_number(fields, number) {
        if field.wire == WIRE_VARINT {
            out.push(field.varint)
        } elif field.wire == WIRE_LEN {
            for value in packed_varints(field) { out.push(value) }
        } else {
            throw ValueError(f"{where} writes field {number} as neither a varint nor a packed run")
        }
    }
    return out
}

fn repeated_floats(fields: list[Field], number: int, where: str) -> list[float] {
    var out: list[float] = []
    for field in fields_by_number(fields, number) {
        if field.wire == WIRE_I32 {
            out.push(as_float(field))
        } elif field.wire == WIRE_LEN {
            for value in packed_floats(field) { out.push(value) }
        } else {
            throw ValueError(f"{where} writes field {number} as neither a float nor a packed run")
        }
    }
    return out
}

fn repeated_doubles(fields: list[Field], number: int, where: str) -> list[float] {
    var out: list[float] = []
    for field in fields_by_number(fields, number) {
        if field.wire == WIRE_I64 {
            out.push(as_double(field))
        } elif field.wire == WIRE_LEN {
            for value in packed_doubles(field) { out.push(value) }
        } else {
            throw ValueError(f"{where} writes field {number} as neither a double nor a packed run")
        }
    }
    return out
}

fn repeated_strings(fields: list[Field], number: int) -> list[str] {
    return [as_string(field) for field in fields_by_number(fields, number)]
}

#: A singular string field, or `""` when the file left it out.
fn optional_string(fields: list[Field], number: int) -> str {
    let found = field_by_number(fields, number)
    if found is null { return "" }
    return as_string(found)
}

#: A singular varint field, or `fallback` when the file left it out.
fn optional_int(fields: list[Field], number: int, fallback: int) -> int {
    let found = field_by_number(fields, number)
    if found is null { return fallback }
    if found.wire != WIRE_VARINT {
        throw ValueError(f"field {number} should be a varint and is not")
    }
    return found.varint
}

#: Where an external tensor says its bytes live, for the error that refuses it.
fn external_location(fields: list[Field]) -> str {
    for entry in fields_by_number(fields, TENSOR_EXTERNAL) {
        let pair = parse_nested(entry)
        if optional_string(pair, ENTRY_KEY) == "location" {
            return optional_string(pair, ENTRY_VALUE)
        }
    }
    return "a file it does not name"
}

#: One `TensorProto`, decoded but not yet uploaded.
class Blob {
    pub let name: str
    pub let shape: list[int]
    pub let values: list[float]

    pub fn init(self, name: str, shape: list[int], values: list[float]) {
        self.name = name
        self.shape = shape
        self.values = values
    }
}

#: The dimensions of a tensor, as the executor can hold them.
#:
#: Two shapes ONNX writes and jaitensor has no room for. A rank-zero scalar
#: becomes `[1]`, which is what the executor's own documentation says a scalar
#: is. A zero-width dimension has nowhere to go at all, and is refused here
#: rather than deeper down where the message would name a tensor the file never
#: mentioned.
fn blob_shape(dims: list[int], where: str) -> list[int] {
    if dims.len() == 0 { return [1] }
    for dimension in dims {
        if dimension < 0 {
            throw ValueError(f"{where} has the negative dimension {dimension} in {dims}")
        }
        if dimension == 0 {
            throw ValueError(f"{where} is empty, at {dims}, and an empty tensor has no place here")
        }
    }
    return dims
}

#: The elements of a tensor stored in `raw_data`.
#:
#: The width check happens before any byte is read, so a truncated or padded
#: run is refused rather than silently producing however many elements the
#: bytes happen to make.
#: The number of elements a shape describes, refusing one that cannot exist.
#:
#: The dimensions come straight out of the file, so their product is attacker
#: controlled. Jaithon's `*` traps on overflow, which would surface as an
#: OverflowError rather than the ValueError this module promises for a
#: malformed model, and a shape that merely fits in an int can still ask for an
#: allocation no machine has. `MAX_ELEMENTS` is far above any real tensor and
#: far below either danger.
let MAX_ELEMENTS = 1 << 34

fn element_count(shape: list[int], where: str) -> int {
    var count = 1
    for dimension in shape {
        if dimension < 0 {
            throw ValueError(f"{where} has dimension {dimension}, and a shape cannot be negative")
        }
        if dimension != 0 and count > MAX_ELEMENTS // dimension {
            throw ValueError(
                f"{where} declares a shape of {shape}, which is more elements than a tensor can hold"
            )
        }
        count *= dimension
    }
    return count
}

fn raw_values(raw: Field, kind: int, count: int, where: str) -> list[float] {
    let width = RAW_WIDTHS.get(kind, 0)
    if width == 0 {
        throw ValueError(f"{where} holds {data_type_name(kind)}, which this reader does not decode")
    }
    let data = as_bytes(raw)
    if data.len() != count * width {
        throw ValueError(
            f"{where} needs {count * width} bytes of {data_type_name(kind)} " +
                f"and its raw_data has {data.len()}"
        )
    }
    if kind == DT_FLOAT { return packed_floats(raw) }
    if kind == DT_DOUBLE { return packed_doubles(raw) }
    if kind == DT_INT32 { return [float(signed_32(value)) for value in packed_fixed32(raw)] }
    if kind == DT_UINT32 { return [float(value) for value in packed_fixed32(raw)] }
    if kind == DT_INT64 { return [float(value) for value in packed_fixed64(raw)] }
    if kind == DT_UINT64 { return [unsigned_64(value) for value in packed_fixed64(raw)] }
    var out: list[float] = []
    for index in 0..count {
        if kind == DT_UINT8 {
            out.push(float(data[index]))
        } elif kind == DT_INT8 {
            let byte = data[index]
            out.push(float(if byte >= 128 { byte - 256 } else { byte }))
        } elif kind == DT_BOOL {
            out.push(if data[index] != 0 { 1.0 } else { 0.0 })
        } elif kind == DT_UINT16 {
            out.push(float(le_u16(data, index * 2)))
        } elif kind == DT_INT16 {
            let half = le_u16(data, index * 2)
            out.push(float(if half >= 0x8000 { half - 0x10000 } else { half }))
        } elif kind == DT_FLOAT16 {
            out.push(f16_from_bits(le_u16(data, index * 2)))
        } elif kind == DT_BFLOAT16 {
            out.push(bf16_from_bits(le_u16(data, index * 2)))
        } else {
            throw ValueError(f"{where} holds {data_type_name(kind)}, which has no raw reading")
        }
    }
    return out
}

#: The elements of a tensor stored in the repeated field its type belongs to.
#:
#: Which field that is is not one per type: everything narrower than a word,
#: and both half-precision formats, ride in `int32_data`, and the two unsigned
#: types share `uint64_data`. A half arrives there as its sixteen bits rather
#: than as a number.
fn typed_values(fields: list[Field], kind: int, where: str) -> list[float] {
    if kind == DT_FLOAT { return repeated_floats(fields, TENSOR_FLOAT_DATA, where) }
    if kind == DT_DOUBLE { return repeated_doubles(fields, TENSOR_DOUBLE_DATA, where) }
    if kind == DT_INT64 {
        return [float(value) for value in repeated_varints(fields, TENSOR_INT64_DATA, where)]
    }
    if kind == DT_UINT32 or kind == DT_UINT64 {
        return [unsigned_64(value) for value in repeated_varints(fields, TENSOR_UINT64_DATA, where)]
    }
    if kind == DT_STRING {
        throw ValueError(f"{where} holds strings, and a network runs on numbers")
    }
    if RAW_WIDTHS.get(kind, 0) == 0 {
        throw ValueError(f"{where} holds {data_type_name(kind)}, which this reader does not decode")
    }
    let stored = repeated_varints(fields, TENSOR_INT32_DATA, where)
    if kind == DT_BOOL { return [if value != 0 { 1.0 } else { 0.0 } for value in stored] }
    if kind == DT_FLOAT16 { return [f16_from_bits(value & 0xFFFF) for value in stored] }
    if kind == DT_BFLOAT16 { return [bf16_from_bits(value & 0xFFFF) for value in stored] }
    return [float(value) for value in stored]
}

#: One `TensorProto`, whichever way it was written.
fn read_blob(field: Field, whose: str) -> Blob {
    let fields = parse_nested(field)
    let name = optional_string(fields, TENSOR_NAME)
    let where = if name.len() > 0 { f"{whose} {name}" } else { whose }
    if field_by_number(fields, TENSOR_SEGMENT) is not null {
        throw ValueError(f"{where} arrives in segments, which only a model past 2 GB is written in")
    }
    if optional_int(fields, TENSOR_LOCATION, 0) == LOCATION_EXTERNAL {
        throw ValueError(
            f"{where} keeps its weights in {external_location(fields)}, " +
                "a file beside the model that this reader does not follow"
        )
    }
    let kind = optional_int(fields, TENSOR_DATA_TYPE, 0)
    let shape = blob_shape(repeated_varints(fields, TENSOR_DIMS, where), where)
    let count = element_count(shape, where)
    let raw = field_by_number(fields, TENSOR_RAW_DATA)
    let values = if raw is not null {
        raw_values(raw, kind, count, where)
    } else {
        typed_values(fields, kind, where)
    }
    if values.len() != count {
        throw ValueError(f"{where} is shaped {shape} and holds {values.len()} values")
    }
    return Blob(name, shape, values)
}

#: The shape a `ValueInfoProto` declares, with `-1` where the file left a
#: symbolic name instead of a number and an empty list where it declared no
#: shape at all.
fn declared_shape(fields: list[Field]) -> list[int] {
    let type_field = field_by_number(fields, VALUE_TYPE)
    if type_field is null { return [] }
    let tensor_field = field_by_number(parse_nested(type_field), TYPE_TENSOR)
    if tensor_field is null { return [] }
    let shape_field = field_by_number(parse_nested(tensor_field), ELEM_SHAPE)
    if shape_field is null { return [] }
    var dims: list[int] = []
    for entry in fields_by_number(parse_nested(shape_field), SHAPE_DIM) {
        let dimension = parse_nested(entry)
        let value = field_by_number(dimension, DIM_VALUE)
        dims.push(if value is null { -1 } else { value.varint })
    }
    return dims
}

#: A model, once read: the graph, and the shapes its inputs and outputs were
#: declared with, which the graph itself has no room for.
class Model {
    pub let graph: Graph
    pub let shapes: dict[str, list[int]]

    pub fn init(self, graph: Graph, shapes: dict[str, list[int]]) {
        self.graph = graph
        self.shapes = shapes
    }
}

#: The reading in progress.
#:
#: It exists to hold the weights, so that a failure halfway through a file
#: releases the tensors already uploaded instead of stranding them on the GPU.
class Reading {
    pub var opset: int
    pub var weights: dict[str, Tensor]
    pub var nodes: list[GraphNode]
    pub var taken: dict[str, bool]
    var minted: int

    pub fn init(self) {
        self.opset = 0
        self.weights = {}
        self.nodes = []
        self.taken = {}
        self.minted = 0
    }

    #: Every tensor read so far, released. Called only on the failing path.
    pub fn discard(self) -> void {
        for (_name, tensor) in self.weights.items() { tensor.free() }
        for node in self.nodes { free_node_tensors(node) }
    }

    #: A name no other value in the file has, for a node this module inserts.
    pub fn mint(self, stem: str) -> str {
        var candidate = stem
        while self.taken.get(candidate, false) {
            self.minted += 1
            candidate = f"{stem}_{self.minted}"
        }
        self.taken[candidate] = true
        return candidate
    }

    #: Take ownership of a tensor and give it a name the graph can read.
    pub fn adopt(self, name: str, tensor: Tensor, where: str) -> void {
        if name.len() == 0 or self.weights.get(name, null) is not null {
            tensor.free()
            if name.len() == 0 {
                throw ValueError(f"{where} has no name, so nothing could read it")
            }
            throw ValueError(f"{where} is the second weight in this file called {name}")
        }
        self.weights[name] = tensor
        self.taken[name] = true
    }

    pub fn hold(self, name: str, shape: list[int], values: list[float], where: str) -> void {
        self.adopt(name, Tensor(shape, values), where)
    }
}

#: Release every tensor a node's attributes carry.
#:
#: A node that is thrown away rather than returned is a node nothing else will
#: ever walk, so whatever it uploaded has to go here.
fn free_node_tensors(node: GraphNode) -> void {
    for (_key, attr) in node.attrs.items() {
        let held = attr.tensor
        if held is not null { held.free() }
    }
}

#: Which of `Attr`'s seven shapes an attribute is in.
#:
#: An exporter old enough to predate the `type` field leaves it out, and then
#: the only evidence is which payload field is present -- which is enough,
#: since each shape has a field number of its own, and only an empty repeat is
#: ambiguous. That one is refused rather than guessed at, because guessing
#: `ints` for what was meant as `floats` writes a field the executor never
#: reads.
fn attribute_kind(fields: list[Field], where: str) -> int {
    let declared = optional_int(fields, ATTR_KIND, KIND_UNDEFINED)
    if declared != KIND_UNDEFINED { return declared }
    if field_by_number(fields, ATTR_FLOATS) is not null { return KIND_FLOATS }
    if field_by_number(fields, ATTR_INTS) is not null { return KIND_INTS }
    if field_by_number(fields, ATTR_STRINGS) is not null { return KIND_STRINGS }
    if field_by_number(fields, ATTR_TENSORS) is not null { return KIND_GRAPH }
    if field_by_number(fields, ATTR_GRAPH) is not null { return KIND_GRAPH }
    if field_by_number(fields, ATTR_GRAPHS) is not null { return KIND_GRAPH }
    if field_by_number(fields, ATTR_SPARSE) is not null { return KIND_GRAPH }
    if field_by_number(fields, ATTR_SPARSES) is not null { return KIND_GRAPH }
    if field_by_number(fields, ATTR_TENSOR) is not null { return KIND_TENSOR }
    if field_by_number(fields, ATTR_STRING) is not null { return KIND_STRING }
    if field_by_number(fields, ATTR_INT) is not null { return KIND_INT }
    if field_by_number(fields, ATTR_FLOAT) is not null { return KIND_FLOAT }
    throw ValueError(f"{where} declares no type and carries no value, so it cannot be read")
}

#: Every attribute of one node.
#:
#: A failure partway through releases the tensors already built, because the
#: node they would have hung off is never returned.
fn read_attributes(fields: list[Field], label: str) -> dict[str, Attr] {
    var attrs: dict[str, Attr] = {}
    try {
        for attribute in fields_by_number(fields, NODE_ATTRIBUTE) {
            let (key, value) = read_attribute(parse_nested(attribute), label)
            if key.len() == 0 { throw ValueError(f"{label} carries an attribute with no name") }
            #: A repeated name would otherwise drop the first attribute on the
            #: floor, and with it the device buffer its tensor had already
            #: uploaded -- unreachable from the graph and never freed.
            if key in attrs {
                throw ValueError(f"{label} carries two attributes named {key}")
            }
            attrs[key] = value
        }
    } catch failure: Error {
        for (_key, attr) in attrs.items() {
            let held = attr.tensor
            if held is not null { held.free() }
        }
        throw failure
    }
    return attrs
}

fn read_attribute(fields: list[Field], label: str) -> tuple[str, Attr] {
    let name = optional_string(fields, ATTR_NAME)
    var where = f"an unnamed attribute of {label}"
    if name.len() > 0 { where = f"the attribute {name} of {label}" }
    if field_by_number(fields, ATTR_REF) is not null {
        throw ValueError(f"{where} refers to a function's argument, and functions are not expanded")
    }
    let kind = attribute_kind(fields, where)
    if kind == KIND_FLOAT {
        let held = field_by_number(fields, ATTR_FLOAT)
        return (name, Attr.of_float(if held is null { 0.0 } else { as_float(held) }))
    }
    if kind == KIND_INT {
        return (name, Attr.of_int(optional_int(fields, ATTR_INT, 0)))
    }
    if kind == KIND_STRING {
        return (name, Attr.of_str(optional_string(fields, ATTR_STRING)))
    }
    if kind == KIND_FLOATS {
        return (name, Attr.of_floats(repeated_floats(fields, ATTR_FLOATS, where)))
    }
    if kind == KIND_INTS {
        return (name, Attr.of_ints(repeated_varints(fields, ATTR_INTS, where)))
    }
    if kind == KIND_STRINGS {
        return (name, Attr.of_strs(repeated_strings(fields, ATTR_STRINGS)))
    }
    if kind == KIND_TENSOR {
        let held = field_by_number(fields, ATTR_TENSOR)
        if held is null { throw ValueError(f"{where} is a tensor attribute holding no tensor") }
        let blob = read_blob(held, where)
        return (name, Attr.of_tensor(Tensor(blob.shape, blob.values)))
    }
    throw ValueError(
        f"{where} is a subgraph or a sparse tensor, which no operator this executor runs takes"
    )
}

#: The tensor a `Constant` node stands for, in whichever of its five shapes the
#: exporter wrote it.
#:
#: For the `value` form this is the attribute's own tensor rather than a copy
#: of it, so the caller must not free that attribute afterwards.
fn constant_tensor(node: GraphNode, label: str) -> Tensor {
    let held = node.attrs.get("value", null)
    if held is not null {
        let stored = held.tensor
        if stored is null { throw ValueError(f"{label} carries a value that is not a tensor") }
        return stored
    }
    let single = node.attrs.get("value_float", null)
    if single is not null { return Tensor([1], [single.f]) }
    let whole = node.attrs.get("value_int", null)
    if whole is not null { return Tensor([1], [float(whole.i)]) }
    let many = node.attrs.get("value_floats", null)
    if many is not null {
        if many.floats.len() == 0 { throw ValueError(f"{label} holds no floats to be a constant") }
        return Tensor([many.floats.len()], many.floats)
    }
    let counts = node.attrs.get("value_ints", null)
    if counts is not null {
        if counts.ints.len() == 0 { throw ValueError(f"{label} holds no ints to be a constant") }
        return Tensor([counts.ints.len()], [float(value) for value in counts.ints])
    }
    throw ValueError(f"{label} has no value attribute, or holds it as a sparse tensor")
}

#: The four operators a pre-13 `Softmax` actually meant.
#:
#: Flatten at the axis, normalise rows of the two-dimensional result, put the
#: shape back. That is the old rule verbatim, and it is also the modern rule
#: whenever the input is already two-dimensional, so nothing is lost by
#: rewriting a file that would have been read correctly either way.
fn legacy_softmax(reading: Reading, node: GraphNode) -> list[GraphNode] {
    if node.inputs.len() != 1 or node.outputs.len() != 1 {
        throw ValueError(f"{node.label()} takes one input and gives one output, and does not here")
    }
    let source = node.inputs[0]
    let target = node.outputs[0]
    let axis = node.attrs.get("axis", null)
    let coerce = if axis is null { 1 } else { axis.i }
    let shape_name = reading.mint(f"{target}_shape")
    let flat_name = reading.mint(f"{target}_flat")
    let normalized = reading.mint(f"{target}_rows")
    #: Nodes are often anonymous in an exported file, and three more anonymous
    #: ones would name nothing in an error, so the value they produce stands in.
    var stem = node.name
    if stem.len() == 0 { stem = target }
    return [
        GraphNode("Shape", name: f"{stem}_shape", inputs: [source], outputs: [shape_name]),
        GraphNode(
            "Flatten",
            name: f"{stem}_flatten",
            inputs: [source],
            outputs: [flat_name],
            attrs: {"axis": Attr.of_int(coerce)}
        ),
        GraphNode(
            node.op,
            name: node.name,
            inputs: [flat_name],
            outputs: [normalized],
            attrs: {"axis": Attr.of_int(1)}
        ),
        GraphNode(
            "Reshape",
            name: f"{stem}_reshape",
            inputs: [normalized, shape_name],
            outputs: [target]
        ),
    ]
}

#: One `NodeProto`, as the nodes it becomes -- usually one, none for a
#: `Constant` that has been folded into the weights, four for a `Softmax` old
#: enough to have meant something else.
fn read_node(reading: Reading, fields: list[Field]) -> list[GraphNode] {
    let op = optional_string(fields, NODE_OP_TYPE)
    let name = optional_string(fields, NODE_NAME)
    if op.len() == 0 {
        var which = "a node in this file"
        if name.len() > 0 { which = f"the node {name}" }
        throw ValueError(f"{which} names no operator")
    }
    let inputs = repeated_strings(fields, NODE_INPUT)
    let outputs = repeated_strings(fields, NODE_OUTPUT)
    var node = GraphNode(op, name: name, inputs: inputs, outputs: outputs)
    let label = node.label()
    let domain = optional_string(fields, NODE_DOMAIN)
    if domain.len() > 0 and domain != "ai.onnx" {
        throw ValueError(f"{label} comes from the {domain} operator set, which is not implemented")
    }
    for (key, value) in read_attributes(fields, label).items() { node.attrs[key] = value }
    try {
        return place_node(reading, node, label)
    } catch failure: Error {
        free_node_tensors(node)
        throw failure
    }
}

#: What becomes of a node once it has been read: folded away, rewritten, or
#: kept.
#:
#: Whatever throws in here, the caller frees the node's attributes -- which is
#: why the tensor a `Constant` hands over is taken out of them first rather
#: than freed beside them. It is either in the weights or it is nowhere, and
#: never in both.
fn place_node(reading: Reading, node: GraphNode, label: str) -> list[GraphNode] {
    let op = node.op
    if op == "Constant" {
        if node.outputs.len() != 1 {
            throw ValueError(
                f"{label} names {node.outputs.len()} outputs and a constant is one value"
            )
        }
        let kept = constant_tensor(node, label)
        let _moved = node.attrs.pop("value", null)
        reading.adopt(node.outputs[0], kept, label)
        free_node_tensors(node)
        return []
    }
    if not is_implemented(op) {
        throw ValueError(f"{label} is not an operator this executor implements")
    }
    #: Reshape took its shape as an attribute until opset 5 and as an input
    #: after it. The executor only reads the input, so the old form becomes a
    #: weight and the node grows the second input it would have had.
    if op == "Reshape" and node.inputs.len() == 1 {
        let shape = node.attrs.get("shape", null)
        if shape is not null {
            let held = reading.mint(f"{node.name}_shape")
            reading.hold(held, [shape.ints.len()], [float(value) for value in shape.ints], label)
            node.inputs.push(held)
        }
    }
    if reading.opset < AXIS_SOFTMAX_OPSET and (op == "Softmax" or op == "LogSoftmax") {
        if reading.opset == 0 {
            throw ValueError(
                f"{label} is in a file that declares no opset, and what a {op} means " +
                    "depends on whether that opset is below 13"
            )
        }
        let rewritten = legacy_softmax(reading, node)
        free_node_tensors(node)
        return rewritten
    }
    return [node]
}

#: The whole file, as a graph and the shapes it declared.
fn import_model(data: bytes) -> Model {
    var fields: list[Field] = []
    try {
        fields = parse_message(data)
    } catch failure: ValueError {
        throw ValueError(f"this file is not a readable ONNX model: {failure.message}")
    }
    let graph_field = field_by_number(fields, MODEL_GRAPH)
    if graph_field is null {
        throw ValueError("this file is not an ONNX model: it holds no graph")
    }

    let reading = Reading()
    for entry in fields_by_number(fields, MODEL_OPSET) {
        let declared = parse_nested(entry)
        let domain = optional_string(declared, OPSET_DOMAIN)
        if domain.len() == 0 or domain == "ai.onnx" {
            reading.opset = optional_int(declared, OPSET_VERSION, reading.opset)
        }
    }

    try {
        return assemble(reading, parse_nested(graph_field))
    } catch failure: Error {
        reading.discard()
        throw failure
    }
}

#: Everything under a `GraphProto`, in the order that lets a name be checked
#: against what came before it.
fn assemble(reading: Reading, fields: list[Field]) -> Model {
    if field_by_number(fields, GRAPH_SPARSE_INITIALIZER) is not null {
        throw ValueError("this model has sparse initializers, which are not implemented")
    }
    for entry in fields_by_number(fields, GRAPH_INITIALIZER) {
        let blob = read_blob(entry, "the initializer")
        reading.hold(blob.name, blob.shape, blob.values, f"the initializer {blob.name}")
    }

    #: Every name the file uses, gathered before a node is rewritten so that an
    #: inserted name cannot collide with one further down the file.
    var node_messages: list[list[Field]] = []
    for entry in fields_by_number(fields, GRAPH_NODE) { node_messages.push(parse_nested(entry)) }
    for node in node_messages {
        for name in repeated_strings(node, NODE_INPUT) { reading.taken[name] = true }
        for name in repeated_strings(node, NODE_OUTPUT) { reading.taken[name] = true }
    }

    var shapes: dict[str, list[int]] = {}
    var inputs: list[str] = []
    for entry in fields_by_number(fields, GRAPH_INPUT) {
        let declared = parse_nested(entry)
        let name = optional_string(declared, VALUE_NAME)
        reading.taken[name] = true
        shapes[name] = declared_shape(declared)
        #: An input the file also gives a weight for is a default rather than
        #: something the caller has to supply, so it is not asked for.
        if reading.weights.get(name, null) is null { inputs.push(name) }
    }
    var outputs: list[str] = []
    for entry in fields_by_number(fields, GRAPH_OUTPUT) {
        let declared = parse_nested(entry)
        let name = optional_string(declared, VALUE_NAME)
        reading.taken[name] = true
        shapes[name] = declared_shape(declared)
        outputs.push(name)
    }
    if outputs.len() == 0 {
        throw ValueError("this model names no outputs, so there is nothing to run it for")
    }

    for message in node_messages {
        for node in read_node(reading, message) { reading.nodes.push(node) }
    }
    let graph = Graph(
        nodes: reading.nodes,
        inputs: inputs,
        outputs: outputs,
        initializers: reading.weights
    )
    #: Ordering the graph now turns a file whose names do not join up into an
    #: error at the point it was read, rather than at the first forward pass.
    let _order = graph.topological()
    return Model(graph, shapes)
}

#: Every tensor an imported graph holds, released.
#:
#: A `Graph` is otherwise plain data, so this is what a program that loads more
#: than one model needs. A tensor that has already been freed ignores being
#: freed again, so calling this twice is harmless -- but running the graph
#: after it is not.
pub fn free_onnx_weights(graph: Graph) -> void {
    for (_name, tensor) in graph.initializers.items() { tensor.free() }
    for node in graph.nodes { free_node_tensors(node) }
}

#: The graph in an ONNX model already in memory.
pub fn read_net_from_onnx_bytes(data: bytes) -> Graph { return import_model(data).graph }

#: The graph in an ONNX model on disk.
#:
#: Raises `ValueError` for a file that is not an ONNX model, for one truncated
#: partway, for a weight this reader cannot decode, and for an operator the
#: executor does not implement -- naming the node in the last case, so that a
#: model can be turned away before a forward pass rather than during one.
pub fn read_net_from_onnx(path: str) -> Graph {
    return read_net_from_onnx_bytes(io.read_bytes(path))
}

#: A network behind `Runnable`, so an ONNX file can stand where OpenCV's
#: detector code puts a `dnn::Net`.
#:
#: One blob goes in and one comes out, which is what that interface is: a model
#: with more than one input or more than one output is refused here and wants
#: `run_graph`, where every name can be given a tensor of its own.
pub class OnnxNet: Runnable {
    #: The imported graph. Free it with `free_onnx_weights` when done.
    pub let graph: Graph
    #: The name the model gives its input, and the shape it declared for it,
    #: with `-1` for a dimension the file left symbolic.
    pub let input_name: str
    pub let input_shape: list[int]
    pub let output_name: str

    pub fn init(self, model: Model) {
        let graph = model.graph
        if graph.inputs.len() != 1 {
            throw ValueError(
                f"this model takes {graph.inputs.len()} inputs, and a Net feeds one blob"
            )
        }
        if graph.outputs.len() != 1 {
            throw ValueError(
                f"this model gives {graph.outputs.len()} outputs, and a Net returns one blob"
            )
        }
        self.graph = graph
        self.input_name = graph.inputs[0]
        self.input_shape = model.shapes.get(self.input_name, [])
        self.output_name = graph.outputs[0]
    }

    #: Run one blob through and return what comes out.
    #:
    #: The blob arrives as the two-dimensional `Mat` `blob_from_image` builds,
    #: which has lost the shape it stands for, so the declared input shape is
    #: what puts it back. A symbolic dimension -- the batch, usually -- is
    #: worked out from how many values actually arrived.
    #:
    #: What comes back is flattened the same way round: the output's first
    #: dimension is the rows and everything after it the columns, so an
    #: `N x C` result is itself and an `N x C x H x W` one is `N` rows of the
    #: rest.
    pub fn run(self, blob: Mat) -> Mat {
        if blob.channels() != 1 {
            throw ValueError(
                f"a blob is one channel and this one has {blob.channels()}; " +
                    "blob_from_image is what lays an image out planar"
            )
        }
        #: The blob is already on the device, so the tensor shares its
        #: storage where it can rather than making a round trip through the
        #: host to say so. A blob that is a view of something larger is copied,
        #: since a tensor wants the whole of what it is given.
        let count = blob.rows * blob.cols
        let fed = if blob.is_continuous() {
            Tensor(self.resolved_shape(count), storage: blob.buffer().view(blob.offset, count))
        } else {
            let values = blob.values()
            Tensor(self.resolved_shape(values.len()), values)
        }
        defer { fed.free() }
        #: `run_best` interprets the first call and compiles the model after,
        #: which is most of the speed for anything run more than once.
        let produced = run_best(self.graph, {self.input_name: fed})
        defer {
            for (_name, tensor) in produced.items() { tensor.free() }
        }
        let output = produced[self.output_name]
        let flat = output.download()
        let rows = output.shape[0]
        return Mat.from_list(rows, flat.len() // rows, CV_MAKETYPE(CV_32F, 1), flat)
    }

    fn resolved_shape(self, count: int) -> list[int] {
        if self.input_shape.len() == 0 {
            throw ValueError(
                f"this model declares no shape for {self.input_name}, so a flat blob " +
                    "cannot be laid out for it"
            )
        }
        var shape = [dimension for dimension in self.input_shape]
        var known = 1
        var symbolic = -1
        for (axis, dimension) in shape.enumerate() {
            if dimension > 0 {
                #: Bounded the same way the initializer shapes are; see
                #: `element_count`. This product is also file-controlled.
                if known > MAX_ELEMENTS // dimension {
                    throw ValueError(
                        f"{self.input_name} is declared {self.input_shape}, which is more elements than a tensor can hold"
                    )
                }
                known *= dimension
                continue
            }
            if dimension == 0 {
                throw ValueError(
                    f"{self.input_name} is declared {self.input_shape}, which is empty"
                )
            }
            if symbolic >= 0 {
                throw ValueError(
                    f"{self.input_name} is declared {self.input_shape}, which leaves more " +
                        "than one dimension for the blob to settle"
                )
            }
            symbolic = axis
        }
        if symbolic < 0 {
            if count != known {
                throw ValueError(f"{self.input_name} is {self.input_shape} and {count} values came")
            }
            return shape
        }
        if known == 0 or count % known != 0 {
            throw ValueError(
                f"{count} values do not divide into {self.input_shape} for {self.input_name}"
            )
        }
        shape[symbolic] = count // known
        return shape
    }
}

#: An ONNX model on disk, behind `Runnable`.
#:
#: Hold on to this rather than the `Net` around it if the weights have to be
#: freed: a `Net` has nowhere to keep a destructor, and this is what knows
#: which graph to hand `free_onnx_weights`.
pub fn read_onnx_engine(path: str) -> OnnxNet {
    return OnnxNet(import_model(io.read_bytes(path)))
}

#: An ONNX model on disk, as the `Net` OpenCV's `readNetFromONNX` returns.
pub fn read_net_from_onnx_as_net(path: str) -> Net { return Net(read_onnx_engine(path)) }
