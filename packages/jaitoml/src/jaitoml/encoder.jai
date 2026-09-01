#: Write TOML documents, byte for byte as Python's `tomli_w` 1.2 does.
#:
#: The layout rules are `tomli_w`'s, kept whole so that a document dumped
#: here and one dumped there compare equal. Every nested dict becomes a
#: `[table]` section after the scalars of its parent; a list of dicts becomes
#: `[[array]]` sections unless every one of them fits an inline table in a
#: hundred columns; every other array is written one element per line with a
#: trailing comma; a key that is not a bare key is quoted like a string.
#:
#: ```
#: dumps({"title": "TOML", "owner": {"name": "Tom"}})
#: #     "title = \"TOML\"\n\n[owner]\nname = \"Tom\"\n"
#: dumps({"xs": [1, 2]})                 # "xs = [\n    1,\n    2,\n]\n"
#: ```
#:
#: Floats print as Jaithon prints them, which is Python's shortest round-trip
#: form, so `1e+20`, `inf` and `nan` come out as `tomli_w` writes them. A
#: `DateTime`, `Date` or `Time` prints its `to_str`. A value of any other class
#: raises `TypeError`, as does a dict key that is not a string.
import std.math as math

from std.io import File
from std.str import StringBuilder
from .datetime import Date, DateTime, Time

#: An inline table longer than this is written as a `[[section]]` instead.
pub const MAX_LINE_LENGTH: int = 100

#: Render `obj`, a dict, as a TOML document.
#:
#: `multiline_strings` writes a string holding a newline as a `"""` block
#: instead of escaping the newline. `indent` is the width of one level of
#: array indentation and must not be negative.
#:
#: Raises `TypeError` for a value TOML cannot hold — a null, a set, an
#: instance of some other class, a key that is not a string — and
#: `ValueError` for a negative indent.
#:
#: ```
#: dumps({"a": {"b": [{"c": 1}]}})
#: #     "[a]\nb = [\n    { c = 1 },\n]\n"
#: dumps({"s": "two\nlines"}, multiline_strings: true)
#: #     "s = \"\"\"\ntwo\nlines\"\"\"\n"
#: ```
pub fn dumps(obj: dict[str, any], multiline_strings: bool = false, indent: int = 4) -> str {
    let ctx = _Context(multiline_strings, indent)
    let out = StringBuilder()
    _table_chunks(obj, ctx, "", false, out)
    return out.build()
}

#: Write `obj` as a TOML document to `fp`, which is open for writing.
#:
#: Takes the same options as `dumps`.
pub fn dump(
    obj: dict[str, any],
    fp: File,
    multiline_strings: bool = false,
    indent: int = 4
) -> void {
    let _ = fp.write(dumps(obj, multiline_strings, indent))
}

class _Context {
    pub let allow_multiline: bool
    pub let indent_str: str

    fn init(self, allow_multiline: bool, indent: int) {
        if indent < 0 { throw ValueError("Indent width must be non-negative") }
        self.allow_multiline = allow_multiline
        self.indent_str = " " * indent
    }
}

fn _table_chunks(
    table: dict[str, any],
    ctx: _Context,
    name: str,
    inside_aot: bool,
    out: StringBuilder
) -> void {
    var yielded = false
    let literals: list[tuple[any, any]] = []
    let tables: list[tuple[any, any, bool]] = []
    for (key, value) in table.items() {
        if isinstance(value, dict) {
            tables.push((key, value, false))
        } elif _is_aot(value) and not _all_suitable_inline(value, ctx) {
            for item in value { tables.push((key, item, true)) }
        } else {
            literals.push((key, value))
        }
    }
    if inside_aot or name.len() > 0 and (literals.len() > 0 or tables.len() == 0) {
        yielded = true
        out.push(inside_aot ? f"[[{name}]]\n" : f"[{name}]\n")
    }
    if literals.len() > 0 {
        yielded = true
        for (key, value) in literals {
            out.push(_format_key_part(key) + " = " + _format_literal(value, ctx, 0) + "\n")
        }
    }
    for (key, value, in_aot) in tables {
        if yielded {
            out.push("\n")
        } else {
            yielded = true
        }
        let key_part = _format_key_part(key)
        let display_name = name.len() > 0 ? name + "." + key_part : key_part
        _table_chunks(value, ctx, display_name, in_aot, out)
    }
}

fn _format_literal(obj: any, ctx: _Context, nest_level: int) -> str {
    if isinstance(obj, bool) { return obj ? "true" : "false" }
    if isinstance(obj, int) { return str(obj) }
    if isinstance(obj, float) { return math.is_nan(obj) ? "nan" : str(obj) }
    if isinstance(obj, str) { return _format_string(obj, ctx.allow_multiline) }
    if isinstance(obj, Date) or isinstance(obj, DateTime) or isinstance(obj, Time) {
        return obj.to_str()
    }
    if isinstance(obj, list) or isinstance(obj, tuple) {
        return _format_inline_array(obj, ctx, nest_level)
    }
    if isinstance(obj, dict) { return _format_inline_table(obj, ctx) }
    throw TypeError(f"Object of type '{_type_name(obj)}' is not TOML serializable")
}

fn _format_inline_table(obj: dict[any, any], ctx: _Context) -> str {
    if obj.len() == 0 { return "{}" }
    let parts = [_format_key_part(key) + " = " + _format_literal(value, ctx, 0) for (key, value) in obj.items()]
    return "{ " + ", ".join(parts) + " }"
}

fn _format_inline_array(obj: any, ctx: _Context, nest_level: int) -> str {
    if len(obj) == 0 { return "[]" }
    let item_indent = ctx.indent_str * (1 + nest_level)
    let closing_indent = ctx.indent_str * nest_level
    let items = [item_indent + _format_literal(item, ctx, nest_level + 1) for item in obj]
    return "[\n" + ",\n".join(items) + ",\n" + closing_indent + "]"
}

fn _format_key_part(part: any) -> str {
    if not isinstance(part, str) {
        throw TypeError(
            f"Invalid mapping key '{part}' of type '{_type_name(part)}'. A string is required."
        )
    }
    if part.len() > 0 and _only_bare_key_chars(part) { return part }
    return _format_string(part, false)
}

fn _format_string(s: str, allow_multiline: bool) -> str {
    let do_multiline = allow_multiline and s.contains("\n")
    let text = do_multiline ? s.replace("\r\n", "\n") : s
    let out = StringBuilder()
    out.push(do_multiline ? "\"\"\"\n" : "\"")
    for char in text {
        if not _is_illegal_basic(char) {
            out.push(char)
            continue
        }
        let compact = _COMPACT_ESCAPES.get(char, "")
        if compact.len() > 0 {
            out.push(do_multiline and char == "\n" ? "\n" : compact)
        } else {
            out.push("\\u" + _hex4(ord(char)))
        }
    }
    out.push(do_multiline ? "\"\"\"" : "\"")
    return out.build()
}

#: A non-empty array whose every element is a dict.
fn _is_aot(obj: any) -> bool {
    if not (isinstance(obj, list) or isinstance(obj, tuple)) { return false }
    if len(obj) == 0 { return false }
    for item in obj {
        if not isinstance(item, dict) { return false }
    }
    return true
}

fn _all_suitable_inline(items: any, ctx: _Context) -> bool {
    for item in items {
        if not _is_suitable_inline_table(item, ctx) { return false }
    }
    return true
}

#: Whether the inline form of `obj` fits on one line of `MAX_LINE_LENGTH`.
fn _is_suitable_inline_table(obj: dict[any, any], ctx: _Context) -> bool {
    let rendered = ctx.indent_str + _format_inline_table(obj, ctx) + ","
    return rendered.len() <= MAX_LINE_LENGTH and not rendered.contains("\n")
}

let _COMPACT_ESCAPES: dict[str, str] = {
    "\x08": "\\b",
    "\n": "\\n",
    "\x0C": "\\f",
    "\r": "\\r",
    "\"": "\\\"",
    "\\": "\\\\",
}

fn _is_illegal_basic(char: str) -> bool {
    if char == "\"" or char == "\\" { return true }
    let code = ord(char)
    return code < 0x20 and code != 0x09 or code == 0x7F
}

fn _only_bare_key_chars(text: str) -> bool {
    for char in text {
        let code = ord(char)
        let bare = code >= 0x61 and code <= 0x7A or
            code >= 0x41 and code <= 0x5A or
            code >= 0x30 and code <= 0x39 or
            char == "-" or
            char == "_"
        if not bare { return false }
    }
    return true
}

fn _type_name(value: any) -> str {
    if value is null { return "NoneType" }
    return type_of(value)
}

fn _hex4(code: int) -> str {
    let digits = "0123456789abcdef"
    var out = ""
    var shift = 12
    while shift >= 0 {
        out = out + digits[code >> shift & 0xF]
        shift -= 4
    }
    return out
}
