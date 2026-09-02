#: Nodes to values: PyYAML's `SafeConstructor`.
#:
#: A scalar becomes null, bool, int, float or str by its tag; a sequence a
#: `list[any]`; a mapping a `dict[any, any]` after its `<<` merge keys are
#: flattened in PyYAML's order -- merged entries first, so the mapping's own
#: entries override them, and a list of mappings merged last to first.
#: `!!set` becomes a set, `!!binary` bytes, `!!omap` and `!!pairs` a list of
#: pairs.
#:
#: Two values fall outside what this language can hold. A mapping key that
#: is null raises `ConstructorError`, since a dict here cannot take one; and
#: an int outside 64 bits becomes a float, where PyYAML would widen to a
#: long. Both are noted in the README.
#:
#: A node reached twice through an alias is constructed once: the value is
#: recorded before its children are built, which is how `&a [*a]` becomes a
#: list containing itself rather than an infinite recursion.
from std.math import INF, NAN
from std.str import to_float, to_int

from .composer import Composer
from .error import ConstructorError, python_repr
from .nodes import Node, NodeKind
from .reader import STRING_NAME
from .resolver import BINARY_TAG, BOOL_TAG, FLOAT_TAG, INT_TAG, MAP_TAG, MERGE_TAG, NULL_TAG
from .resolver import OMAP_TAG, PAIRS_TAG, SEQ_TAG, SET_TAG, STR_TAG, is_float_text

let BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

#: Builds the value of each document in a text.
pub class Constructor {
    #: The composer the nodes come from.
    pub let composer: Composer

    var constructed: dict[int, any]
    var recursive: dict[int, bool]

    #: Start constructing `text`.
    pub fn init(self, text: str, name: str = STRING_NAME) {
        self.composer = Composer(text, name)
        self.constructed = {}
        self.recursive = {}
    }

    #: Whether another document follows.
    pub fn check_data(self) -> bool { return self.composer.check_node() }

    #: The value of the next document; null once the stream is exhausted, and
    #: also for a document that is itself null.
    pub fn get_data(self) -> any {
        if self.composer.check_node() {
            let node = self.composer.get_node()
            if node is not null { return self.construct_document(node) }
        }
        return null
    }

    #: The value of the only document, or null for an empty stream.
    pub fn get_single_data(self) -> any {
        let node = self.composer.get_single_node()
        if node is not null { return self.construct_document(node) }
        return null
    }

    #: The value of `node`, with the per-document tables reset afterwards.
    pub fn construct_document(self, node: Node) -> any {
        let data = self.construct_object(node)
        self.constructed = {}
        self.recursive = {}
        return data
    }

    #: The value of `node`, constructing it once however often it is reached.
    pub fn construct_object(self, node: Node) -> any {
        let key = id(node)
        if key in self.constructed { return self.constructed[key] }
        if key in self.recursive {
            throw ConstructorError(
                null,
                null,
                "found unconstructable recursive node",
                node.start_mark
            )
        }
        self.recursive[key] = true
        let data = self.construct_by_tag(node)
        self.recursive.remove(key)
        return data
    }

    fn remember(self, node: Node, data: any) -> void { self.constructed[id(node)] = data }

    fn construct_by_tag(self, node: Node) -> any {
        let tag = node.tag
        if tag == NULL_TAG {
            self.construct_scalar(node)
            self.remember(node, null)
            return null
        }
        if tag == BOOL_TAG {
            let value = self.construct_yaml_bool(node)
            self.remember(node, value)
            return value
        }
        if tag == INT_TAG {
            let value = self.construct_yaml_int(node)
            self.remember(node, value)
            return value
        }
        if tag == FLOAT_TAG {
            let value = self.construct_yaml_float(node)
            self.remember(node, value)
            return value
        }
        if tag == STR_TAG {
            let value = self.construct_scalar(node)
            self.remember(node, value)
            return value
        }
        if tag == BINARY_TAG {
            let value = self.construct_yaml_binary(node)
            self.remember(node, value)
            return value
        }
        if tag == SEQ_TAG { return self.construct_yaml_seq(node) }
        if tag == MAP_TAG { return self.construct_yaml_map(node) }
        if tag == SET_TAG { return self.construct_yaml_set(node) }
        if tag == OMAP_TAG { return self.construct_yaml_pairs(node, "an ordered map") }
        if tag == PAIRS_TAG { return self.construct_yaml_pairs(node, "pairs") }
        throw ConstructorError(
            null,
            null,
            f"could not determine a constructor for the tag {python_repr(tag)}",
            node.start_mark
        )
    }

    #: The text of a scalar node; a mapping with a `=` value key yields that
    #: value's text, as in PyYAML.
    pub fn construct_scalar(self, node: Node) -> str {
        if node.kind != NodeKind.Scalar {
            throw ConstructorError(
                null,
                null,
                f"expected a scalar node, but found {node.id()}",
                node.start_mark
            )
        }
        return node.value
    }

    #: The children of a sequence node, constructed.
    pub fn construct_sequence(self, node: Node) -> list[any] {
        if node.kind != NodeKind.Sequence {
            throw ConstructorError(
                null,
                null,
                f"expected a sequence node, but found {node.id()}",
                node.start_mark
            )
        }
        return [self.construct_object(child) for child in node.items]
    }

    #: The pairs of a mapping node as a dict, merge keys flattened first.
    pub fn construct_mapping(self, node: Node) -> dict[any, any] {
        if node.kind != NodeKind.Mapping {
            throw ConstructorError(
                null,
                null,
                f"expected a mapping node, but found {node.id()}",
                node.start_mark
            )
        }
        self.flatten_mapping(node)
        let mapping: dict[any, any] = {}
        self.fill_mapping(node, mapping)
        return mapping
    }

    fn fill_mapping(self, node: Node, mapping: dict[any, any]) -> void {
        for (key_node, value_node) in node.pairs {
            let key = self.construct_object(key_node)
            if key is null {
                throw ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found a null key, which a dict cannot hold",
                    key_node.start_mark
                )
            }
            if isinstance(key, list) or isinstance(key, dict) or isinstance(key, set) {
                throw ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark
                )
            }
            let value = self.construct_object(value_node)
            mapping[key] = value
        }
    }

    #: Replace every `<<` entry of `node` by the entries it merges, in
    #: PyYAML's order.
    pub fn flatten_mapping(self, node: Node) -> void {
        let merge: list[tuple[Node, Node]] = []
        var index = 0
        while index < node.pairs.len() {
            let (key_node, value_node) = node.pairs[index]
            if key_node.tag == MERGE_TAG {
                node.pairs.remove_at(index)
                if value_node.kind == NodeKind.Mapping {
                    self.flatten_mapping(value_node)
                    merge.extend(value_node.pairs)
                } elif value_node.kind == NodeKind.Sequence {
                    let submerge: list[list[tuple[Node, Node]]] = []
                    for subnode in value_node.items {
                        if subnode.kind != NodeKind.Mapping {
                            throw ConstructorError(
                                "while constructing a mapping",
                                node.start_mark,
                                f"expected a mapping for merging, but found {subnode.id()}",
                                subnode.start_mark
                            )
                        }
                        self.flatten_mapping(subnode)
                        submerge.push(subnode.pairs)
                    }
                    submerge.reverse()
                    for pairs in submerge { merge.extend(pairs) }
                } else {
                    throw ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"expected a mapping or list of mappings for merging, but found {value_node.id()}",
                        value_node.start_mark
                    )
                }
            } else {
                index += 1
            }
        }
        if merge.len() > 0 {
            let combined: list[tuple[Node, Node]] = []
            combined.extend(merge)
            combined.extend(node.pairs)
            node.pairs = combined
        }
    }

    fn construct_yaml_bool(self, node: Node) -> bool {
        let value = self.construct_scalar(node).lower()
        if value == "true" { return true }
        if value == "false" { return false }
        throw ConstructorError(
            null,
            null,
            f"found {python_repr(node.value)}, which is not a core-schema bool",
            node.start_mark
        )
    }

    fn construct_yaml_int(self, node: Node) -> any {
        var value = self.construct_scalar(node).strip().replace("_", "")
        var sign = 1
        if value.len() > 0 and value[0] == "-" {
            sign = -1
            value = value[1:]
        } elif value.len() > 0 and value[0] == "+" {
            value = value[1:]
        }
        var base = 10
        var digits = value
        if value.starts_with("0x") {
            base = 16
            digits = value[2:]
        } elif value.starts_with("0o") {
            base = 8
            digits = value[2:]
        }
        if digits.len() == 0 or not all_digits(digits, base) {
            throw ConstructorError(
                null,
                null,
                f"found {python_repr(node.value)}, which is not a core-schema int",
                node.start_mark
            )
        }
        let parsed = to_int(digits, base)
        if parsed is not null { return sign * parsed }
        return float(sign) * widen(digits, base)
    }

    fn construct_yaml_float(self, node: Node) -> float {
        var value = self.construct_scalar(node).strip().replace("_", "").lower()
        var sign = 1.0
        if value.len() > 0 and value[0] == "-" {
            sign = -1.0
            value = value[1:]
        } elif value.len() > 0 and value[0] == "+" {
            value = value[1:]
        }
        if value == ".inf" { return sign * INF }
        if value == ".nan" { return NAN }
        let numeric = is_float_text(value) or value.len() > 0 and all_digits(value, 10)
        if not numeric or value.ends_with("nan") {
            throw ConstructorError(
                null,
                null,
                f"found {python_repr(node.value)}, which is not a core-schema float",
                node.start_mark
            )
        }
        if value.starts_with(".") { value = "0" + value }
        if value.ends_with(".") { value = value + "0" }
        let point = value.find(".e")
        if point >= 0 { value = value[0:point] + ".0" + value[point + 1:] }
        let parsed = to_float(value)
        if parsed is null {
            throw ConstructorError(
                null,
                null,
                f"found {python_repr(node.value)}, which is not a core-schema float",
                node.start_mark
            )
        }
        return sign * parsed
    }

    fn construct_yaml_binary(self, node: Node) -> bytes {
        let text = self.construct_scalar(node)
        let decoded = base64_decode(text)
        if decoded is null {
            throw ConstructorError(null, null, "failed to decode base64 data", node.start_mark)
        }
        return decoded
    }

    fn construct_yaml_seq(self, node: Node) -> list[any] {
        let data: list[any] = []
        self.remember(node, data)
        data.extend(self.construct_sequence(node))
        return data
    }

    fn construct_yaml_map(self, node: Node) -> dict[any, any] {
        let data: dict[any, any] = {}
        self.remember(node, data)
        if node.kind != NodeKind.Mapping {
            throw ConstructorError(
                null,
                null,
                f"expected a mapping node, but found {node.id()}",
                node.start_mark
            )
        }
        self.flatten_mapping(node)
        self.fill_mapping(node, data)
        return data
    }

    fn construct_yaml_set(self, node: Node) -> set[any] {
        let data: set[any] = set()
        self.remember(node, data)
        for key in self.construct_mapping(node).keys() { data.add(key) }
        return data
    }

    fn construct_yaml_pairs(self, node: Node, what: str) -> list[any] {
        let pairs: list[any] = []
        self.remember(node, pairs)
        if node.kind != NodeKind.Sequence {
            throw ConstructorError(
                f"while constructing {what}",
                node.start_mark,
                f"expected a sequence, but found {node.id()}",
                node.start_mark
            )
        }
        for subnode in node.items {
            if subnode.kind != NodeKind.Mapping {
                throw ConstructorError(
                    f"while constructing {what}",
                    node.start_mark,
                    f"expected a mapping of length 1, but found {subnode.id()}",
                    subnode.start_mark
                )
            }
            if subnode.pairs.len() != 1 {
                throw ConstructorError(
                    f"while constructing {what}",
                    node.start_mark,
                    f"expected a single mapping item, but found {subnode.pairs.len()} items",
                    subnode.start_mark
                )
            }
            let (key_node, value_node) = subnode.pairs[0]
            let key = self.construct_object(key_node)
            let value = self.construct_object(value_node)
            pairs.push((key, value))
        }
        return pairs
    }
}

fn digit_value(ch: str) -> int {
    let code = ord(ch)
    if code >= 0x30 and code <= 0x39 { return code - 0x30 }
    if code >= 0x61 and code <= 0x66 { return code - 0x61 + 10 }
    if code >= 0x41 and code <= 0x46 { return code - 0x41 + 10 }
    return -1
}

fn all_digits(digits: str, base: int) -> bool {
    for ch in digits {
        let value = digit_value(ch)
        if value < 0 or value >= base { return false }
    }
    return true
}

#: `digits` in `base` as a float, for magnitudes an int cannot hold.
#:
#: Decimal goes through the correctly rounded parser; hex and octal
#: accumulate, which rounds once per digit past fifty-three bits.
fn widen(digits: str, base: int) -> float {
    if base == 10 { return to_float(digits) ?? 0.0 }
    var value = 0.0
    for ch in digits { value = value * float(base) + float(digit_value(ch)) }
    return value
}

#: Decode base64 the way Python's `base64.decodebytes` does: whitespace and
#: other non-alphabet characters are skipped, and null is returned for a
#: length that no padding can complete.
pub fn base64_decode(text: str) -> bytes? {
    let values: list[int] = []
    for ch in text {
        if ch == "=" { continue }
        let index = BASE64.find(ch)
        if index < 0 { continue }
        values.push(index)
    }
    if values.len() % 4 == 1 { return null }
    let out: list[int] = []
    var index = 0
    while index + 4 <= values.len() {
        let a = values[index]
        let b = values[index + 1]
        let c = values[index + 2]
        let d = values[index + 3]
        out.push((a << 2 | b >> 4) & 0xFF)
        out.push((b << 4 | c >> 2) & 0xFF)
        out.push((c << 6 | d) & 0xFF)
        index += 4
    }
    let rest = values.len() - index
    if rest == 2 {
        out.push((values[index] << 2 | values[index + 1] >> 4) & 0xFF)
    } elif rest == 3 {
        out.push((values[index] << 2 | values[index + 1] >> 4) & 0xFF)
        out.push((values[index + 1] << 4 | values[index + 2] >> 2) & 0xFF)
    }
    return bytes(out)
}

#: Encode `data` the way Python's `base64.encodebytes` does: lines of 76
#: characters, each ending in a newline.
pub fn base64_encode(data: bytes) -> str {
    var out = ""
    var line = ""
    var index = 0
    let length = data.len()
    while index < length {
        let a = data[index]
        let b = index + 1 < length ? data[index + 1] : 0
        let c = index + 2 < length ? data[index + 2] : 0
        var quad = BASE64[a >> 2] + BASE64[(a & 0x3) << 4 | b >> 4]
        quad = quad + (index + 1 < length ? BASE64[(b & 0xF) << 2 | c >> 6] : "=")
        quad = quad + (index + 2 < length ? BASE64[c & 0x3F] : "=")
        line = line + quad
        if line.len() >= 76 {
            out = out + line + "\n"
            line = ""
        }
        index += 3
    }
    if line.len() > 0 { out = out + line + "\n" }
    return out
}
