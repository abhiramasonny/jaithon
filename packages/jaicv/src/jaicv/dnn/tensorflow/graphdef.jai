#: The graph: one `NodeDef`, its inputs, and the attributes hung off it.
from ..protobuf import Field, as_bool, as_float, as_string, field_by_number
from ..protobuf import fields_by_number, parse_nested
from .tensors import read_shape_proto, read_tensor_proto, repeated_varints
from .tensors import TfTensor

#: `GraphDef.node`.
pub let GRAPH_NODE = 1

#: `NodeDef`. Field 4 is the device the node was placed on, which inference
#: on this machine has no use for.
let NODE_NAME = 1
let NODE_OP = 2
let NODE_INPUT = 3
let NODE_ATTR = 5

#: `NodeDef.attr` is a `map<string, AttrValue>`, and protobuf encodes a map as
#: a repeated message of key and value.
let ENTRY_KEY = 1
let ENTRY_VALUE = 2

#: `AttrValue`. Exactly one of these is set on any given attribute. Field 9 is
#: a placeholder name and field 10 a function reference, neither of which
#: survives freezing.
let ATTR_LIST = 1
let ATTR_S = 2
let ATTR_I = 3
let ATTR_F = 4
let ATTR_B = 5
let ATTR_TYPE = 6
let ATTR_SHAPE = 7
pub let ATTR_TENSOR = 8

#: `AttrValue.ListValue` numbers its repeats the way `AttrValue` numbers its
#: singulars, and the only list this importer reads is of integers.
let LIST_I = 3

#: One `NodeDef`, with its input names already canonicalised.
#:
#: A TensorFlow input is `node`, `node:k` for the `k`-th output of a node with
#: several, or `^node` for a control dependency. Control edges carry no value
#: and are dropped here: they order side effects, and there are none left in a
#: frozen inference graph. `node:0` is the same name as `node`, and anything
#: past `:0` is kept verbatim so that a graph reading a second output fails by
#: naming a value nothing produces rather than by silently reading the first.
pub class TfNode {
    pub let name: str
    pub let op: str
    pub let inputs: list[str]
    pub let attrs: dict[str, list[Field]]

    pub fn init(self, name: str, op: str, inputs: list[str], attrs: dict[str, list[Field]]) {
        self.name = name
        self.op = op
        self.inputs = inputs
        self.attrs = attrs
    }

    pub fn label(self) -> str { return f"the {self.op} node {self.name}" }
}

fn canonical_input(raw: str) -> str? {
    if raw.starts_with("^") { return null }
    let colon = raw.rfind(":")
    if colon < 0 { return raw }
    let index = raw.slice(colon + 1).to_int()
    if index is not null and index == 0 { return raw.slice(0, colon) }
    return raw
}

pub fn read_node(fields: list[Field], index: int) -> TfNode {
    let name_field = field_by_number(fields, NODE_NAME)
    let name = if name_field is null { f"node{index}" } else { as_string(name_field) }
    let op_field = field_by_number(fields, NODE_OP)
    if op_field is null { throw ValueError(f"the node {name} names no operator") }
    var inputs: list[str] = []
    for entry in fields_by_number(fields, NODE_INPUT) {
        let cleaned = canonical_input(as_string(entry))
        if cleaned is not null { inputs.push(cleaned) }
    }
    var attrs: dict[str, list[Field]] = {}
    for entry in fields_by_number(fields, NODE_ATTR) {
        let pair = parse_nested(entry)
        let key = field_by_number(pair, ENTRY_KEY)
        let value = field_by_number(pair, ENTRY_VALUE)
        if key is null { continue }
        attrs[as_string(key)] = if value is null { [] } else { parse_nested(value) }
    }
    return TfNode(name, as_string(op_field), inputs, attrs)
}

pub fn attr_of(node: TfNode, name: str) -> list[Field]? { return node.attrs.get(name, null) }

pub fn attr_string(node: TfNode, name: str, fallback: str) -> str {
    let found = attr_of(node, name)
    if found is null { return fallback }
    let value = field_by_number(found, ATTR_S)
    if value is null { return fallback }
    return as_string(value)
}

pub fn attr_int(node: TfNode, name: str, fallback: int) -> int {
    let found = attr_of(node, name)
    if found is null { return fallback }
    let value = field_by_number(found, ATTR_I)
    if value is null { return fallback }
    return value.varint
}

pub fn attr_float(node: TfNode, name: str, fallback: float) -> float {
    let found = attr_of(node, name)
    if found is null { return fallback }
    let value = field_by_number(found, ATTR_F)
    if value is null { return fallback }
    return as_float(value)
}

pub fn attr_bool(node: TfNode, name: str, fallback: bool) -> bool {
    let found = attr_of(node, name)
    if found is null { return fallback }
    let value = field_by_number(found, ATTR_B)
    if value is null { return fallback }
    return as_bool(value)
}

pub fn attr_type(node: TfNode, name: str, fallback: int) -> int {
    let found = attr_of(node, name)
    if found is null { return fallback }
    let value = field_by_number(found, ATTR_TYPE)
    if value is null { return fallback }
    return value.varint
}

#: An `AttrValue` holding a `ListValue` of integers, which is how every
#: per-axis attribute -- `strides`, `ksize`, `dilations`, `squeeze_dims` --
#: arrives.
pub fn attr_ints(node: TfNode, name: str) -> list[int]? {
    let found = attr_of(node, name)
    if found is null { return null }
    let listed = field_by_number(found, ATTR_LIST)
    if listed is null { return null }
    return repeated_varints(parse_nested(listed), LIST_I, node.label())
}

#: The `TensorShapeProto` a `Placeholder` declares, or null when it declares
#: nothing this reader can use.
pub fn attr_shape(node: TfNode, name: str) -> list[int]? {
    let found = attr_of(node, name)
    if found is null { return null }
    let value = field_by_number(found, ATTR_SHAPE)
    if value is null { return null }
    return read_shape_proto(parse_nested(value), node.label())
}

pub fn attr_tensor(node: TfNode, name: str) -> TfTensor? {
    let found = attr_of(node, name)
    if found is null { return null }
    let value = field_by_number(found, ATTR_TENSOR)
    if value is null { return null }
    return read_tensor_proto(parse_nested(value), value.data.len(), node.label())
}
