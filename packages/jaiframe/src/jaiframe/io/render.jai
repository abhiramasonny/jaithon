#: A column's rows as text: the number and timestamp spellings every writer
#: agrees on, and the two host passes that produce them.
import std.io as sio
import std.str as text_ops
from std.math import INF, is_nan
from std.time import DateTime

from ..column import Column, Dtype
from ..column import f32_bits, wide_unpack
from .kernels import blanked, blanked_wide

#: The template `str.format` needs for a caller's float format.
#:
#: Three spellings are accepted because three are in use: a whole template
#: (`"{:.2f}"`), a bare spec (`".2f"`), and the printf form pandas takes
#: (`"%.2f"`). They all mean the same thing and none of them is worth making a
#: caller look up.
fn float_template(spec: str) -> str {
    if spec.contains("{") { return spec }
    if spec.len() > 0 and spec[0] == "%" { return "{:" + spec[1:spec.len()] + "}" }
    return "{:" + spec + "}"
}

#: The shortest decimal that reads back as the same float32.
#:
#: A `Float64` column is a float32 lane, so the value that comes off the device
#: is a float64 holding a float32: 0.1 downloads as 0.10000000149011612 and
#: printing that is neither what the writer put in nor what a reader wants.
#: Nine significant digits always round-trip a float32 and six sometimes do, so
#: the answer is found rather than assumed — the first precision whose text
#: parses back to the same bit pattern wins, which is one attempt for the
#: values a data file is actually full of and eight for a third.
#:
#: The result always carries a `.` or an exponent, even when the value is
#: integral: a float column that wrote `38` would read back as `Int64`, and a
#: round trip that changes the dtype is not one.
fn shortest_float(value: float) -> str {
    if value != value { return "NaN" }
    if value == INF { return "Infinity" }
    if value == -INF { return "-Infinity" }
    let bits = f32_bits(value)
    var digits = 1
    while digits <= 9 {
        let rendered = ("{:." + f"{digits}" + "g}").format(value)
        let parsed = text_ops.to_float(rendered)
        if parsed is not null and f32_bits(parsed ?? 0.0) == bits { return decimalize(rendered) }
        digits += 1
    }
    return decimalize(f"{value}")
}

#: `38` is an integer and `38.0` is a float, and a file has to say which.
fn decimalize(rendered: str) -> str {
    for at in 0..rendered.len() {
        let character = rendered[at]
        if character == "." or character == "e" or character == "E" { return rendered }
    }
    return rendered + ".0"
}

#: A timestamp as ISO 8601, which is the only spelling every reader agrees on.
#:
#: Floor division, not truncation, so a moment before the epoch lands on the
#: second below it and its millisecond remainder stays positive — the same
#: split `wide_pack` makes, for the same reason.
fn iso_stamp(millis: int) -> str {
    let second = millis // 1000
    let remainder = millis - second * 1000
    return DateTime(second, nanosecond: remainder * 1000000).to_iso()
}

#: One rendered cell of a non-dictionary column, given the lane values.
fn plain_cell(dtype: Dtype, value: float, template: str?) -> str {
    if dtype == Dtype.Bool { return if value != 0.0 { "true" } else { "false" } }
    let spec = template
    if spec is not null { return (spec ?? "").format(value) }
    return shortest_float(value)
}

#: One field, quoted the way `std.io.format_csv` quotes it.
#:
#: Routed through `format_csv` on a one-cell table rather than reimplementing
#: the rule, so a file this module writes and `std.io.parse_csv` can never
#: disagree about what needed a quote.
pub fn csv_field(cell: str, delimiter: str) -> str {
    let rendered = sio.format_csv([[cell]], delimiter)
    return rendered[0:rendered.len() - 1]
}

#: Whether a rendered cell of this dtype can go out unquoted.
#:
#: A quote is a property of the characters a rendering can contain, and for a
#: number those are known ahead of the data. Checking the delimiter against
#: them once per column is what keeps the writer from scanning ten million
#: rendered numbers for a comma none of them holds.
fn safe_without_quotes(dtype: Dtype, delimiter: str) -> bool {
    let risky = match dtype {
        Dtype.Bool => "truefals",
        Dtype.Datetime => "0123456789-:.+TZ",
        _ => "0123456789-+.eEInfityaN",
    }
    return not risky.contains(delimiter)
}

#: Every row of a column as text, missing rows as `na_rep`.
#:
#: This is the O(rows) host pass the writers cannot avoid, and it is the only
#: one they make: one download per lane feeds it, and nothing inside it touches
#: the device.
#:
#: `quote_for` names the delimiter when the text is going into a CSV. Quoting
#: then happens once per dictionary level rather than once per row — a text
#: column has as many distinct strings as it has levels however tall it is, and
#: whether a string needs a quote is a property of the string.
pub fn rendered_column(
    column: Column,
    na_rep: str,
    float_format: str?,
    quote_for: str? = null
) -> list[str] {
    let length = column.length
    if length == 0 { return [] }
    let template = if float_format is null { null } else { float_template(float_format ?? "") }
    let delimiter = quote_for
    let blank = if delimiter is null { na_rep } else { csv_field(na_rep, delimiter ?? ",") }
    if column.dtype.is_dictionary() {
        var levels = column.levels ?? []
        if delimiter is not null {
            levels = levels.map(|level| csv_field(level, delimiter ?? ","))
        }
        let codes = blanked(column.values, column.valid, length)
        var out: list[str] = []
        for code in codes {
            if is_nan(code) {
                out.push(blank)
                continue
            }
            let at = int(code)
            out.push(if at >= 0 and at < levels.len() { levels[at] } else { blank })
        }
        return out
    }
    #: A caller's own template may pad or prefix, so its output is checked cell
    #: by cell; the built-in renderings are not.
    let bare = delimiter is null or
        template is null and safe_without_quotes(column.dtype, delimiter ?? ",")
    if column.dtype.is_wide() {
        let (low, high) = blanked_wide(column)
        let stamped = column.dtype == Dtype.Datetime
        var out: list[str] = []
        for (row, slot) in high.enumerate() {
            if is_nan(slot) {
                out.push(blank)
                continue
            }
            let exact = wide_unpack(low[row], slot)
            let cell = if stamped { iso_stamp(exact) } else { f"{exact}" }
            out.push(if bare { cell } else { csv_field(cell, delimiter ?? ",") })
        }
        return out
    }
    var out: list[str] = []
    for value in blanked(column.values, column.valid, length) {
        if is_nan(value) {
            out.push(blank)
            continue
        }
        let cell = plain_cell(column.dtype, value, template)
        out.push(if bare { cell } else { csv_field(cell, delimiter ?? ",") })
    }
    return out
}

#: The rows `start..start + count` of a column, as text.
#:
#: Display asks for a head and a tail and nothing between them, so this
#: downloads exactly the slice it was asked for. Printing a ten-million-row
#: frame moves sixty values per lane, not ten million.
pub fn rendered_slice(
    column: Column,
    start: int,
    count: int,
    na_rep: str,
    float_format: str?
) -> list[str] {
    if count <= 0 { return [] }
    let template = if float_format is null { null } else { float_template(float_format ?? "") }
    let mask = column.valid
    let present = if mask is null { [] } else { mask.download(start, count) }
    let masked = present.len() > 0
    if column.dtype.is_dictionary() {
        let levels = column.levels ?? []
        let codes = column.values.download(start, count)
        var out: list[str] = []
        for (row, code) in codes.enumerate() {
            if masked and present[row] == 0.0 {
                out.push(na_rep)
                continue
            }
            let at = int(code)
            out.push(if at >= 0 and at < levels.len() { levels[at] } else { na_rep })
        }
        return out
    }
    if column.dtype.is_wide() {
        let low = column.values.download(start, count)
        let high = (column.wide ?? column.values).download(start, count)
        let stamped = column.dtype == Dtype.Datetime
        var out: list[str] = []
        for (row, slot) in low.enumerate() {
            if masked and present[row] == 0.0 {
                out.push(na_rep)
                continue
            }
            let exact = wide_unpack(slot, high[row])
            out.push(if stamped { iso_stamp(exact) } else { f"{exact}" })
        }
        return out
    }
    let values = column.values.download(start, count)
    var out: list[str] = []
    for (row, value) in values.enumerate() {
        if masked and present[row] == 0.0 or is_nan(value) {
            out.push(na_rep)
            continue
        }
        out.push(plain_cell(column.dtype, value, template))
    }
    return out
}
