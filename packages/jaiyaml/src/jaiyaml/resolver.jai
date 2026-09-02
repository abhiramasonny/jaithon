#: Tag resolution for untagged nodes: the YAML 1.2 core schema.
#:
#: A plain scalar with no tag gets one from its text: `null`, `Null`,
#: `NULL`, `~` and the empty string are null; `true`/`True`/`TRUE` and
#: `false`/`False`/`FALSE` are bool; decimal, `0o` octal and `0x` hex are
#: int; a decimal with a point or exponent, `.inf` in its three spellings
#: with an optional sign, and `.nan` in three spellings are float; `<<` is
#: the merge key; everything else is str. Quoted scalars are always str,
#: and untagged collections are seq and map.
#:
#: This is where the package deliberately parts from PyYAML, which
#: implements YAML 1.1: there `yes`, `no`, `on` and `off` are bools, `010`
#: is octal, `0o17` and `1e5` are strings, `1:30` is sexagesimal, `1_000`
#: is an int, and `2001-12-14` is a timestamp. All of those are plain
#: strings here (except `010`, which is the decimal ten). The resolver is
#: shared by the loader and the dumper, so the dumper quotes exactly the
#: strings the loader would otherwise misread.
#:
#: ```
#: resolve(NodeKind.Scalar, "0x1F", true)     # "tag:yaml.org,2002:int"
#: resolve(NodeKind.Scalar, "yes", true)      # "tag:yaml.org,2002:str"
#: resolve(NodeKind.Scalar, "12", false)      # "tag:yaml.org,2002:str"
#: ```
from .nodes import NodeKind

#: The tag every untagged quoted scalar, and every unmatched plain one, gets.
pub const DEFAULT_SCALAR_TAG: str = "tag:yaml.org,2002:str"

#: The tag an untagged sequence gets.
pub const DEFAULT_SEQUENCE_TAG: str = "tag:yaml.org,2002:seq"

#: The tag an untagged mapping gets.
pub const DEFAULT_MAPPING_TAG: str = "tag:yaml.org,2002:map"

pub const NULL_TAG: str = "tag:yaml.org,2002:null"
pub const BOOL_TAG: str = "tag:yaml.org,2002:bool"
pub const INT_TAG: str = "tag:yaml.org,2002:int"
pub const FLOAT_TAG: str = "tag:yaml.org,2002:float"
pub const STR_TAG: str = "tag:yaml.org,2002:str"
pub const SEQ_TAG: str = "tag:yaml.org,2002:seq"
pub const MAP_TAG: str = "tag:yaml.org,2002:map"
pub const SET_TAG: str = "tag:yaml.org,2002:set"
pub const MERGE_TAG: str = "tag:yaml.org,2002:merge"
pub const BINARY_TAG: str = "tag:yaml.org,2002:binary"
pub const OMAP_TAG: str = "tag:yaml.org,2002:omap"
pub const PAIRS_TAG: str = "tag:yaml.org,2002:pairs"

#: The tag for a node of `kind` whose text is `value`.
#:
#: `implicit` is whether the node was a plain scalar with no tag: only then
#: does the text decide. Collections ignore `value`.
pub fn resolve(kind: NodeKind, value: str?, implicit: bool) -> str {
    if kind == NodeKind.Sequence { return DEFAULT_SEQUENCE_TAG }
    if kind == NodeKind.Mapping { return DEFAULT_MAPPING_TAG }
    if not implicit { return DEFAULT_SCALAR_TAG }
    return resolve_plain(value ?? "")
}

#: The core-schema tag of a plain scalar's text.
pub fn resolve_plain(text: str) -> str {
    if is_null_text(text) { return NULL_TAG }
    if is_bool_text(text) { return BOOL_TAG }
    if is_int_text(text) { return INT_TAG }
    if is_float_text(text) { return FLOAT_TAG }
    if text == "<<" { return MERGE_TAG }
    return DEFAULT_SCALAR_TAG
}

#: Whether `text` is one of the core schema's null spellings.
pub fn is_null_text(text: str) -> bool {
    return text == "" or text == "~" or text == "null" or text == "Null" or text == "NULL"
}

#: Whether `text` is one of the core schema's bool spellings.
pub fn is_bool_text(text: str) -> bool {
    return text == "true" or
        text == "True" or
        text == "TRUE" or
        text == "false" or
        text == "False" or
        text == "FALSE"
}

fn is_decimal_digit(ch: str) -> bool { return ch >= "0" and ch <= "9" }

fn is_octal_digit(ch: str) -> bool { return ch >= "0" and ch <= "7" }

fn is_hex_digit(ch: str) -> bool {
    return ch >= "0" and ch <= "9" or ch >= "a" and ch <= "f" or ch >= "A" and ch <= "F"
}

fn all_digits(text: str, start: int, accept: fn(str) -> bool) -> bool {
    if start >= text.len() { return false }
    var index = start
    while index < text.len() {
        if not accept(text[index]) { return false }
        index += 1
    }
    return true
}

#: Whether `text` is a core-schema int: `[-+]?[0-9]+`, `0o[0-7]+` or
#: `0x[0-9a-fA-F]+`.
pub fn is_int_text(text: str) -> bool {
    if text.len() == 0 { return false }
    if text.starts_with("0o") { return all_digits(text, 2, is_octal_digit) }
    if text.starts_with("0x") { return all_digits(text, 2, is_hex_digit) }
    var start = 0
    if text[0] == "-" or text[0] == "+" { start = 1 }
    return all_digits(text, start, is_decimal_digit)
}

#: Whether `text` is a core-schema float: a decimal with a point or an
#: exponent, `.inf` with an optional sign, or `.nan`.
pub fn is_float_text(text: str) -> bool {
    if text.len() == 0 { return false }
    var index = 0
    if text[0] == "-" or text[0] == "+" { index = 1 }
    let rest = text[index:]
    if rest == ".inf" or rest == ".Inf" or rest == ".INF" { return true }
    if index == 0 and (rest == ".nan" or rest == ".NaN" or rest == ".NAN") { return true }
    let length = text.len()
    var digits_before = 0
    while index < length and is_decimal_digit(text[index]) {
        index += 1
        digits_before += 1
    }
    var digits_after = 0
    var has_point = false
    if index < length and text[index] == "." {
        has_point = true
        index += 1
        while index < length and is_decimal_digit(text[index]) {
            index += 1
            digits_after += 1
        }
    }
    if digits_before == 0 and digits_after == 0 { return false }
    if digits_before == 0 and not has_point { return false }
    var has_exponent = false
    if index < length and (text[index] == "e" or text[index] == "E") {
        has_exponent = true
        index += 1
        if index < length and (text[index] == "-" or text[index] == "+") { index += 1 }
        var exponent_digits = 0
        while index < length and is_decimal_digit(text[index]) {
            index += 1
            exponent_digits += 1
        }
        if exponent_digits == 0 { return false }
    }
    if index != length { return false }
    return has_point or has_exponent
}
