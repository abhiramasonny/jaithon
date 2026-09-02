#: Values to nodes: PyYAML's `SafeRepresenter`.
#:
#: null, bool, int, float and str become scalars with the core-schema tags;
#: a list or tuple a sequence; a dict a mapping; a set a `!!set` mapping of
#: nulls; bytes a `!!binary` literal block. Anything else -- a class
#: instance, a function -- raises `RepresenterError`, as PyYAML's safe
#: dumper does.
#:
#: A container reached twice through the same object is represented once
#: and aliased the second time, which is how PyYAML writes `&id001` and
#: `*id001`; scalars are never aliased. Floats are written with the
#: shortest round-tripping digits, lower-cased, and an exponent form that
#: lacks a point gains `.0` before the `e`, so `1e+16` becomes `1.0e+16`.
#:
#: `sort_keys` sorts a mapping's keys the way Python would sort its items:
#: strings by code point, numbers by value, and not at all when the keys
#: are of kinds Python cannot compare, such as a str beside an int, since
#: PyYAML swallows that `TypeError` and keeps insertion order.
from std.math import is_inf, is_nan

from .constructor import base64_encode
from .error import RepresenterError
from .nodes import Node, NodeKind
from .resolver import BINARY_TAG, BOOL_TAG, FLOAT_TAG, INT_TAG, MAP_TAG, NULL_TAG, SEQ_TAG
from .resolver import SET_TAG, STR_TAG

#: Builds the node graph of one value.
pub class Representer {
    let default_style: str?
    let default_flow_style: bool?
    let sort_keys: bool
    var represented: dict[int, Node]
    var alias_key: int?

    #: A representer with PyYAML's three representation options.
    pub fn init(
        self,
        default_style: str? = null,
        default_flow_style: bool? = false,
        sort_keys: bool = true
    ) {
        self.default_style = default_style
        self.default_flow_style = default_flow_style
        self.sort_keys = sort_keys
        self.represented = {}
        self.alias_key = null
    }

    #: The node graph of `data`, with the alias table reset afterwards.
    pub fn represent(self, data: any) -> Node {
        let node = self.represent_data(data)
        self.represented = {}
        self.alias_key = null
        return node
    }

    #: The node for `data`, or the node already made for the same object.
    pub fn represent_data(self, data: any) -> Node {
        if ignore_aliases(data) {
            self.alias_key = null
        } else {
            self.alias_key = id(data)
        }
        let key = self.alias_key
        if key is not null and (key ?? 0) in self.represented { return self.represented[key ?? 0] }
        if data is null { return self.represent_scalar(NULL_TAG, "null") }
        if isinstance(data, bool) {
            return self.represent_scalar(BOOL_TAG, data ? "true" : "false")
        }
        if isinstance(data, int) { return self.represent_scalar(INT_TAG, str(data)) }
        if isinstance(data, float) { return self.represent_float(data) }
        if isinstance(data, str) { return self.represent_scalar(STR_TAG, data) }
        if isinstance(data, bytes) {
            return self.represent_scalar(BINARY_TAG, base64_encode(data), "|")
        }
        if isinstance(data, list) or isinstance(data, tuple) {
            return self.represent_sequence(SEQ_TAG, data)
        }
        if isinstance(data, dict) { return self.represent_mapping(MAP_TAG, data) }
        if isinstance(data, set) {
            let value: dict[any, any] = {}
            for item in data { value[item] = null }
            return self.represent_mapping(SET_TAG, value)
        }
        throw RepresenterError(f"cannot represent an object: {repr(data)}")
    }

    #: A scalar node, remembered for aliasing when the value can be aliased.
    pub fn represent_scalar(self, tag: str, value: str, style: str? = null) -> Node {
        let node = Node.scalar(tag, value, null, null, style ?? self.default_style)
        let key = self.alias_key
        if key is not null { self.represented[key ?? 0] = node }
        return node
    }

    #: A sequence node; flow style is chosen as PyYAML chooses it.
    pub fn represent_sequence(self, tag: str, sequence: any, flow_style: bool? = null) -> Node {
        let node = Node.sequence(tag, [], null, null, flow_style)
        let key = self.alias_key
        if key is not null { self.represented[key ?? 0] = node }
        var best_style = true
        for item in sequence {
            let node_item = self.represent_data(item)
            if not (node_item.kind == NodeKind.Scalar and node_item.style is null) {
                best_style = false
            }
            node.items.push(node_item)
        }
        if flow_style is null {
            if self.default_flow_style is not null {
                node.flow_style = self.default_flow_style
            } else {
                node.flow_style = best_style
            }
        }
        return node
    }

    #: A mapping node; keys are sorted when `sort_keys` asks and Python could.
    pub fn represent_mapping(
        self,
        tag: str,
        mapping: dict[any, any],
        flow_style: bool? = null
    ) -> Node {
        let node = Node.mapping(tag, [], null, null, flow_style)
        let key = self.alias_key
        if key is not null { self.represented[key ?? 0] = node }
        var best_style = true
        var keys = mapping.keys()
        if self.sort_keys { keys = python_sorted(keys) ?? keys }
        for item_key in keys {
            let node_key = self.represent_data(item_key)
            let node_value = self.represent_data(mapping[item_key])
            if not (node_key.kind == NodeKind.Scalar and node_key.style is null) {
                best_style = false
            }
            if not (node_value.kind == NodeKind.Scalar and node_value.style is null) {
                best_style = false
            }
            node.pairs.push((node_key, node_value))
        }
        if flow_style is null {
            if self.default_flow_style is not null {
                node.flow_style = self.default_flow_style
            } else {
                node.flow_style = best_style
            }
        }
        return node
    }

    fn represent_float(self, data: float) -> Node {
        var value = ""
        if is_nan(data) {
            value = ".nan"
        } elif is_inf(data) {
            value = data > 0.0 ? ".inf" : "-.inf"
        } else {
            value = str(data).lower()
            if "." not in value and "e" in value {
                let at = value.find("e")
                value = value[0:at] + ".0" + value[at:]
            }
        }
        return self.represent_scalar(FLOAT_TAG, value)
    }
}

fn ignore_aliases(data: any) -> bool {
    if data is null { return true }
    if isinstance(data, str) or isinstance(data, bytes) or isinstance(data, bool) { return true }
    if isinstance(data, int) or isinstance(data, float) { return true }
    if isinstance(data, tuple) and len(data) == 0 { return true }
    return false
}

#: Python's ordering of two keys: negative, zero or positive, or null when
#: Python would raise `TypeError` comparing them.
pub fn python_compare(a: any, b: any) -> int? {
    let a_number = isinstance(a, int) or isinstance(a, float) or isinstance(a, bool)
    let b_number = isinstance(b, int) or isinstance(b, float) or isinstance(b, bool)
    if a_number and b_number {
        let x = as_number(a)
        let y = as_number(b)
        if x < y { return -1 }
        if x > y { return 1 }
        return 0
    }
    if isinstance(a, str) and isinstance(b, str) {
        if a < b { return -1 }
        if a > b { return 1 }
        return 0
    }
    if isinstance(a, tuple) and isinstance(b, tuple) {
        var index = 0
        while index < len(a) and index < len(b) {
            let order = python_compare(a[index], b[index])
            if order is null { return null }
            if (order ?? 0) != 0 { return order }
            index += 1
        }
        if len(a) < len(b) { return -1 }
        if len(a) > len(b) { return 1 }
        return 0
    }
    return null
}

fn as_number(value: any) -> float {
    if isinstance(value, bool) { return value ? 1.0 : 0.0 }
    return float(value)
}

#: `keys` in Python's sorted order, or null when Python's `sorted` would
#: raise, which PyYAML answers by leaving the order alone.
pub fn python_sorted(keys: list[any]) -> list[any]? {
    if keys.len() < 2 { return keys }
    var failed = false
    let compare = fn(a: any, b: any) -> int {
        let order = python_compare(a, b)
        if order is null {
            failed = true
            return 0
        }
        return order
    }
    let sorted_keys = merge_sort(keys, compare)
    if failed { return null }
    return sorted_keys
}

fn merge_sort(items: list[any], compare: fn(any, any) -> int) -> list[any] {
    if items.len() <= 1 { return items }
    let middle = items.len() // 2
    let left = merge_sort(items[0:middle], compare)
    let right = merge_sort(items[middle:], compare)
    let merged: list[any] = []
    var i = 0
    var j = 0
    while i < left.len() and j < right.len() {
        if compare(right[j], left[i]) < 0 {
            merged.push(right[j])
            j += 1
        } else {
            merged.push(left[i])
            i += 1
        }
    }
    while i < left.len() {
        merged.push(left[i])
        i += 1
    }
    while j < right.len() {
        merged.push(right[j])
        j += 1
    }
    return merged
}
