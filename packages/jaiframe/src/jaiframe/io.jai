#: A frame as text: CSV and JSON on disk, and the printed table on screen.
#:
#: Every path here crosses the boundary between a device allocation and host
#: text, and the whole design is about crossing it once. A reader parses each
#: field into a per-column host lane and hands the finished lane to
#: `Buffer.upload` in a single call; a writer pulls each column down in a
#: single `download` and formats from that. Reaching for `Column.get` or
#: `Column.set` a cell at a time instead would be a device round trip per cell,
#: and a million-row file has enough cells to make that the whole program
#: rather than a detail of it.
#:
#: What is left on the host after that is the formatting itself, and it stays
#: there: the output of a writer is a host string, so turning ten million
#: values into ten million pieces of text is work no dispatch can take away.
#: The device does the two things that are reductions rather than renderings —
#: folding a validity lane into its payload so the writer makes one pass
#: instead of two, and counting the bytes a dictionary column's strings occupy
#: for `memory_usage(deep: true)`.
#:
#: Display never reads the whole frame. `to_string` picks the head and tail
#: rows it is going to show and downloads exactly those slices, so printing a
#: ten-million-row frame costs the same as printing a sixty-row one.
#:
#: ```
#: let frame = read_csv("sales.csv", parse_dates: ["day"])
#: defer { frame.free() }
#: print(to_string(frame))
#: to_csv(frame, "out.csv", float_format: ".2f")
#: to_json(frame, "out.json", orient: "split")     # dtypes and nulls survive
#: ```
import std.fmt as fmt
import std.io as sio
import std.json as sjson
import std.str as text_ops
from std.gpu import Buffer, Kernel
from std.math import INF, NAN, is_nan
from std.time import DateTime

from .column import Column, DataFrame, Dtype, Index, Series, Value
from .column import f32_bits, wide_unpack
from .construct import column_from_values, infer_dtype

#: The strings a reader treats as a missing cell when `keep_default_na` is on.
#:
#: A file's own convention wins through `na_values`; this is the vocabulary
#: spreadsheets and databases already emit. `""` is in the list because an
#: empty field in a numeric column is a gap, not a zero, and the R and SQL
#: spellings are here because a CSV rarely says which tool wrote it.
pub let DEFAULT_NA_WORDS = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
    "NIL",
    "nil",
]

#: How many rows `to_string` prints before it truncates to a head and a tail.
pub const DEFAULT_MAX_ROWS = 60

#: How many columns `to_string` prints before it elides the middle ones.
pub const DEFAULT_MAX_COLS = 20

const BYTES_GROUP = 256
const BYTES_GROUPS = 256
const LEVEL_TILE = 1024

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint BYTES_GROUP = 256;
constexpr constant uint BYTES_GROUPS = 256;
constexpr constant uint LEVEL_TILE = 1024;

/* A missing row, folded into the payload lane so a writer's host pass has one
 * question to ask per cell instead of two. */
kernel void io_blank_nulls(device const float* src [[buffer(0)]],
                           device const float* valid [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (valid[gid] == 0.0f) ? NAN : src[gid];
}

/* The same four rows at a time, for a column whose length divides by four.
 * The kernel does nothing but move eight bytes in and four out, so a quarter
 * of the threads is very nearly a quarter of the cost. */
kernel void io_blank_nulls4(device const float4* src [[buffer(0)]],
                            device const float4* valid [[buffer(1)]],
                            device float4* out [[buffer(2)]],
                            constant uint& quads [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= quads) return;
    out[gid] = select(float4(NAN), src[gid], valid[gid] != 0.0f);
}

/* How many bytes a dictionary column's strings occupy, counting a level once
 * per row that uses it.
 *
 * The rows are codes, so this is a gather out of a table with one entry per
 * level, and neighbouring threads hit the same handful of entries over and
 * over -- ten million rows over eight levels would read eight addresses ten
 * million times. The table is staged in threadgroup memory once per group and
 * read from there; past LEVEL_TILE levels the staging would cost more than it
 * saves and the gather goes straight to device memory instead.
 *
 * The fold runs in `long` because the answer does not fit a float32. Ten
 * million rows of ten bytes is 10^8 and float32 counts integers exactly only
 * to 2^24, so each group reports its total as the same two-lane split an
 * Int64 column stores. */
kernel void io_text_bytes(device const float* codes [[buffer(0)]],
                          device const float* valid [[buffer(1)]],
                          device const float* lengths [[buffer(2)]],
                          device float* low [[buffer(3)]],
                          device float* high [[buffer(4)]],
                          constant uint& count [[buffer(5)]],
                          constant uint& levels [[buffer(6)]],
                          constant uint& has_valid [[buffer(7)]],
                          uint gid [[thread_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float table[LEVEL_TILE];
    threadgroup long scratch[BYTES_GROUP];
    bool staged = levels <= LEVEL_TILE;
    if (staged) {
        for (uint at = lid; at < levels; at += BYTES_GROUP) table[at] = lengths[at];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long total = 0;
    for (uint row = gid; row < count; row += BYTES_GROUP * BYTES_GROUPS) {
        if (has_valid != 0u && valid[row] == 0.0f) continue;
        uint code = (uint)codes[row];
        if (code >= levels) continue;
        total += (long)(staged ? table[code] : lengths[code]);
    }
    scratch[lid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = BYTES_GROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        long sum = scratch[0];
        long hi = sum >> 24;
        low[wid] = (float)(sum - (hi << 24));
        high[wid] = (float)hi;
    }
}
"""

var _kernels: dict[str, Kernel] = {}

fn compiled(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, entry)
    _kernels[entry] = kernel
    return kernel
}

fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

#: Every row of a plain lane as a host float, missing rows already NaN.
#:
#: `Column.to_floats` answers the same question by downloading the validity
#: lane alongside the payload and patching the result row by row. Folding the
#: mask into the payload on the device instead is one dispatch, one download
#: and no second host pass, which on a wide frame is one avoided pass per
#: column rather than one in total.
fn blanked(lane: Buffer, valid: Buffer?, length: int) -> list[float] {
    if length == 0 { return [] }
    let mask = valid
    if mask is null { return lane.download() }
    let out = Buffer(length)
    if length % 4 == 0 {
        compiled("io_blank_nulls4").dispatch_async([lane, mask, out], length // 4, [length // 4])
    } else {
        compiled("io_blank_nulls").dispatch_async([lane, mask, out], length, [length])
    }
    let values = out.download()
    out.free()
    return values
}

#: The two lanes of a wide column, with the high half NaN where a row is
#: missing.
#:
#: The value has to come back exact, so the low lane is downloaded untouched
#: and recombined on the host. Only the high half carries the missing flag,
#: which costs the writer one `is_nan` where reading a separate mask would
#: cost it a third download and a third pass.
fn blanked_wide(column: Column) -> (list[float], list[float]) {
    if column.length == 0 { return ([], []) }
    let high = column.wide ?? column.values
    return (column.values.download(), blanked(high, column.valid, column.length))
}

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
fn csv_field(cell: str, delimiter: str) -> str {
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
fn rendered_column(
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
fn rendered_slice(
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

#: The words this reader will treat as a missing cell.
fn na_lookup(na_values: list[str]?, keep_default_na: bool) -> dict[str, bool] {
    var words: dict[str, bool] = {}
    if keep_default_na {
        for word in DEFAULT_NA_WORDS { words[word] = true }
    }
    for word in na_values ?? [] { words[word] = true }
    return words
}

#: The integer a field spells, or `null` when it does not spell one.
#:
#: Deliberately stricter than `to_int`: a field is an integer only when every
#: character after an optional sign is a digit, so `1e6` and `0x10` stay for
#: the float and text passes to claim. An integer past the 48 bits a column
#: stores is not one either, and falls through to `Float64` rather than
#: raising — a reader that refused a large number would be worse than one that
#: rounds it.
fn integer_of(field: str) -> int? {
    let body = field.strip()
    var at = 0
    if body.len() > 0 and (body[0] == "-" or body[0] == "+") { at = 1 }
    if at >= body.len() { return null }
    while at < body.len() {
        let character = body[at]
        if character < "0" or character > "9" { return null }
        at += 1
    }
    return text_ops.to_int(body)
}

#: The float a field spells, or `null`.
fn float_of(field: str) -> float? { return text_ops.to_float(field.strip()) }

#: The boolean a field spells, or `null`.
#:
#: Only the three spellings of the words. `1` and `0` are left to the integer
#: pass on purpose: a column of counts that happens to hold only ones and
#: zeroes is a column of counts.
fn bool_of(field: str) -> bool? {
    return match field.strip() {
        "true" | "True" | "TRUE" => true,
        "false" | "False" | "FALSE" => false,
        _ => null,
    }
}

#: The host value a field of text spells.
#:
#: The order is the order of the ladder: the word spellings of truth first,
#: then a strict integer, then anything else numeric, then the text itself.
fn literal_of(field: str) -> any {
    let truth = bool_of(field)
    if truth is not null { return truth ?? false }
    let whole = integer_of(field)
    if whole is not null { return whole ?? 0 }
    let number = float_of(field)
    if number is not null { return number ?? 0.0 }
    return field
}

#: The dtype a column of text fields wants.
#:
#: The ladder that settles `Bool` beside `Int64` lives in `construct.jai` and
#: is shared with every other way of building a frame, so this pass does not
#: repeat it: it walks the fields once, and hands `infer_dtype` one witness for
#: each family it saw. Four boxed values decide the column instead of ten
#: million, and the two readers cannot drift apart. Text ends the walk, since
#: nothing found after it can promote a column past `Str`.
fn sniff_dtype(fields: list[str], words: dict[str, bool]) -> Dtype {
    var witnesses: list[any] = []
    var saw_bool = false
    var saw_int = false
    var saw_float = false
    for field in fields {
        if words.get(field, false) { continue }
        let literal = literal_of(field)
        if isinstance(literal, bool) {
            if saw_bool { continue }
            saw_bool = true
        } elif isinstance(literal, int) {
            if saw_int { continue }
            saw_int = true
        } elif isinstance(literal, float) {
            if saw_float { continue }
            saw_float = true
        } else {
            witnesses.push(literal)
            break
        }
        witnesses.push(literal)
    }
    return infer_dtype(witnesses)
}

#: One column, built from the field text of one CSV column.
#:
#: This is the bulk conversion the reader is built around. Every field is
#: parsed into a host lane list and the whole lane reaches the device in the
#: single `Buffer.upload` inside the matching `Column.from_*` constructor.
#: The obvious alternative — allocate the column and `Column.set` each cell —
#: is a device round trip per cell, which on a million-row file is seconds of
#: wall clock for a file that reads in well under one.
#:
#: A field that will not parse as the dtype asks raises rather than quietly
#: becoming a null: inference never produces a dtype its own column cannot
#: hold, so a failure here means the caller named a `dtypes` entry the data
#: does not honour, and that is worth hearing about.
fn column_from_fields(
    name: str,
    fields: list[str],
    dtype: Dtype,
    words: dict[str, bool]
) -> Column {
    if dtype == Dtype.Datetime { return datetime_column(name, fields, words) }
    if dtype.is_dictionary() {
        var texts: list[str?] = []
        for field in fields { texts.push(if words.get(field, false) { null } else { field }) }
        return Column.from_strs(texts, dtype)
    }
    var nulls: list[bool] = []
    if dtype == Dtype.Bool {
        var truths: list[bool] = []
        for field in fields {
            if words.get(field, false) {
                truths.push(false)
                nulls.push(true)
                continue
            }
            let truth = bool_of(field)
            if truth is null { throw refusal(name, field, dtype) }
            truths.push(truth ?? false)
            nulls.push(false)
        }
        return Column.from_bools(truths, nulls)
    }
    if dtype == Dtype.Int64 {
        var wholes: list[int] = []
        for field in fields {
            if words.get(field, false) {
                wholes.push(0)
                nulls.push(true)
                continue
            }
            let whole = integer_of(field)
            if whole is null { throw refusal(name, field, dtype) }
            wholes.push(whole ?? 0)
            nulls.push(false)
        }
        return Column.from_ints(wholes, nulls)
    }
    var numbers: list[float] = []
    for field in fields {
        if words.get(field, false) {
            numbers.push(NAN)
            nulls.push(true)
            continue
        }
        let number = float_of(field)
        if number is null { throw refusal(name, field, dtype) }
        numbers.push(number ?? NAN)
        nulls.push(false)
    }
    return Column.from_floats(numbers, nulls)
}

fn refusal(name: str, field: str, dtype: Dtype) -> ValueError {
    return ValueError(f"column {name} was asked for {dtype.name()} and holds {repr(field)}")
}

#: A `Datetime` column from field text.
#:
#: Every rule about what a date looks like belongs to `timeseries.jai`, so the
#: fields go there whole rather than being picked apart here. The import is
#: local because the calendar is the only thing this reader borrows from that
#: module and a `read_csv` without `parse_dates` has no use for it.
fn datetime_column(name: str, fields: list[str], words: dict[str, bool]) -> Column {
    from .timeseries import to_datetime
    var texts: list[str?] = []
    for field in fields { texts.push(if words.get(field, false) { null } else { field }) }
    return to_datetime(texts)
}

#: Records split into fields, following RFC 4180.
#:
#: `std.io.parse_csv` is the split for an ordinary file, and it fixes the quote
#: at `"` because RFC 4180 does. A caller who names another one gets the same
#: walk with that character; the branch exists rather than a pre-translation
#: because rewriting the quotes first would corrupt any literal `"` the data
#: holds.
fn split_rows(text: str, delimiter: str, quotechar: str) -> list[list[str]] {
    if quotechar == "\"" { return sio.parse_csv(text, delimiter) }
    if quotechar.len() != 1 {
        throw ValueError(f"quotechar must be one character, got {repr(quotechar)}")
    }
    if quotechar == delimiter or quotechar == "\n" or quotechar == "\r" {
        throw ValueError(f"quotechar may not be the delimiter or a newline, got {repr(quotechar)}")
    }
    var rows: list[list[str]] = []
    var row: list[str] = []
    var field: list[str] = []
    var quoted = false
    var started = false
    var at = 0
    while at < text.len() {
        let character = text[at]
        at += 1
        if quoted {
            if character != quotechar {
                field.push(character)
                continue
            }
            if at < text.len() and text[at] == quotechar {
                field.push(quotechar)
                at += 1
                continue
            }
            quoted = false
            continue
        }
        if character == quotechar and field.len() == 0 {
            quoted = true
            started = true
            continue
        }
        if character == delimiter {
            row.push("".join(field))
            field = []
            started = true
            continue
        }
        if character == "\n" or character == "\r" {
            if character == "\r" and at < text.len() and text[at] == "\n" { at += 1 }
            row.push("".join(field))
            rows.push(row)
            field = []
            row = []
            started = false
            continue
        }
        field.push(character)
        started = true
    }
    if quoted { throw ParseError("unterminated quoted field at the end of the CSV input") }
    if started {
        row.push("".join(field))
        rows.push(row)
    }
    return rows
}

fn numbered(width: int) -> list[str] {
    var names: list[str] = []
    for at in 0..width { names.push(f"{at}") }
    return names
}

#: Where the columns named by `index_col` sit.
fn index_positions(spec: any, names: list[str]) -> list[int] {
    if spec is null { return [] }
    if isinstance(spec, list) {
        var out: list[int] = []
        for entry in spec { out.push(one_index_position(entry, names)) }
        return out
    }
    return [one_index_position(spec, names)]
}

fn one_index_position(spec: any, names: list[str]) -> int {
    if isinstance(spec, str) {
        for (at, name) in names.enumerate() {
            if name == spec { return at }
        }
        throw KeyError(f"index_col names {spec}, which is not a column in this file")
    }
    if isinstance(spec, int) {
        let at = if spec < 0 { names.len() + spec } else { spec }
        if at < 0 or at >= names.len() {
            throw IndexError(f"index_col {spec} is outside the {names.len()} columns of this file")
        }
        return at
    }
    throw TypeError(f"index_col takes a column name or a position, not {repr(spec)}")
}

#: Move the index columns out of the frame's columns and into its index.
fn lift_index(
    names: list[str],
    columns: list[Column],
    positions: list[int]
) -> (list[str], list[Column], Index?) {
    if positions.len() == 0 { return (names, columns, null) }
    var taken: dict[int, bool] = {}
    var levels: list[Column] = []
    var labels: list[str?] = []
    for at in positions {
        if taken.get(at, false) { throw ValueError(f"index_col names column {at} twice") }
        taken[at] = true
        levels.push(columns[at])
        labels.push(names[at])
    }
    var rest_names: list[str] = []
    var rest_columns: list[Column] = []
    for (at, name) in names.enumerate() {
        if taken.get(at, false) { continue }
        rest_names.push(name)
        rest_columns.push(columns[at])
    }
    return (rest_names, rest_columns, Index.of(levels, labels))
}

#: Read delimited text as a frame.
#:
#: The split is `std.io.parse_csv` — RFC 4180 quoting, `\n` or `\r\n` records —
#: and everything after it is this module's: each column's fields are parsed
#: into a host lane and uploaded in one call.
#:
#: A column's dtype is inferred from its own fields unless `dtypes` names one.
#: `parse_dates` asks for `Datetime`, which is the same thing as naming it in
#: `dtypes`, spelled the way pandas spells it. `na_values` adds to the words in
#: `DEFAULT_NA_WORDS`; `keep_default_na` turns those off and leaves only the
#: caller's.
#:
#: `header` says whether the first surviving record names the columns.
#: `names` supplies or replaces those names. `skiprows` drops records before
#: the header is looked for, which is how a file with a preamble is read, and
#: `nrows` stops after that many data records. A record shorter than the header
#: is padded with missing cells; a longer one is a malformed file and raises.
#:
#: ```
#: read_csv_text("a,b\n1,x\n2,y\n")            # int64 and str, a range index
#: read_csv_text("a,b\n1,x\n", index_col: "a") # one column under an int64 index
#: ```
pub fn read_csv_text(
    text: str,
    delimiter: str = ",",
    header: bool = true,
    names: list[str]? = null,
    index_col: any = null,
    dtypes: dict[str, any]? = null,
    na_values: list[str]? = null,
    keep_default_na: bool = true,
    parse_dates: list[str]? = null,
    skiprows: int = 0,
    nrows: int = -1,
    quotechar: str = "\""
) -> DataFrame {
    if skiprows < 0 { throw ValueError(f"skiprows cannot be {skiprows}") }
    let records = split_rows(text, delimiter, quotechar)
    var at = if skiprows < records.len() { skiprows } else { records.len() }
    var column_names = names ?? []
    if header and at < records.len() {
        let heading = records[at]
        at += 1
        if names is null { column_names = heading }
    }
    if column_names.len() == 0 and at < records.len() {
        column_names = numbered(records[at].len())
    }
    let width = column_names.len()
    var lanes: list[list[str]] = []
    for _column in 0..width { lanes.push([]) }
    var taken = 0
    while at < records.len() {
        if nrows >= 0 and taken >= nrows { break }
        let record = records[at]
        at += 1
        if record.len() > width {
            throw ValueError(
                f"record {taken} has {record.len()} fields where the header names {width}"
            )
        }
        for (position, lane) in lanes.enumerate() {
            lane.push(if position < record.len() { record[position] } else { "" })
        }
        taken += 1
    }
    let words = na_lookup(na_values, keep_default_na)
    var wanted_dates: dict[str, bool] = {}
    for name in parse_dates ?? [] { wanted_dates[name] = true }
    var columns: list[Column] = []
    for (position, name) in column_names.enumerate() {
        let named = (dtypes ?? {}).get(name, null)
        var dtype = if named is null { Dtype.Float64 } else { as_dtype(named) }
        if named is null {
            dtype = if wanted_dates.get(name, false) {
                Dtype.Datetime
            } else {
                sniff_dtype(lanes[position], words)
            }
        }
        columns.push(column_from_fields(name, lanes[position], dtype, words))
    }
    let (rest_names, rest_columns, index) = lift_index(
        column_names,
        columns,
        index_positions(index_col, column_names)
    )
    return DataFrame(rest_names, rest_columns, index: index ?? Index.range(taken))
}

#: Read the delimited file at `path` as a frame. The arguments are
#: `read_csv_text`'s.
pub fn read_csv(
    path: str,
    delimiter: str = ",",
    header: bool = true,
    names: list[str]? = null,
    index_col: any = null,
    dtypes: dict[str, any]? = null,
    na_values: list[str]? = null,
    keep_default_na: bool = true,
    parse_dates: list[str]? = null,
    skiprows: int = 0,
    nrows: int = -1,
    quotechar: str = "\""
) -> DataFrame {
    return read_csv_text(
        sio.read_file(path),
        delimiter: delimiter,
        header: header,
        names: names,
        index_col: index_col,
        dtypes: dtypes,
        na_values: na_values,
        keep_default_na: keep_default_na,
        parse_dates: parse_dates,
        skiprows: skiprows,
        nrows: nrows,
        quotechar: quotechar
    )
}

fn as_dtype(spec: any) -> Dtype {
    if isinstance(spec, Dtype) { return spec }
    if isinstance(spec, str) { return Dtype.from_name(spec) }
    throw TypeError(f"{repr(spec)} does not name a dtype")
}

#: Render a frame as delimited text.
#:
#: `index` writes the index as leading columns, named as its levels are named;
#: `header` writes the column names as the first record. `na_rep` is what a
#: missing cell becomes, empty by default, which is what `read_csv` reads back
#: as missing. `float_format` is a `str.format` spec — `".2f"`, `"{:.2f}"` or
#: `"%.2f"` — and without one a float is written as the shortest decimal that
#: reads back as the same float32.
#:
#: One flat list of pieces and one join at the end, rather than a string per
#: record: joining once allocates the finished file once, where `out = out +
#: record` allocates and copies the whole of it again for every row.
pub fn to_csv_text(
    frame: DataFrame,
    delimiter: str = ",",
    header: bool = true,
    index: bool = true,
    na_rep: str = "",
    float_format: str? = null
) -> str {
    #: `format_csv` owns which delimiters are legal, and a frame whose every
    #: field turns out to need no quote would otherwise never ask it.
    csv_field("", delimiter)
    let gutter = if index { frame.index.materialize() } else { [] }
    defer {
        for level in gutter { level.free() }
    }
    var titles: list[str] = []
    if index {
        for (at, name) in frame.index.names.enumerate() {
            titles.push(csv_field(name ?? f"index_{at}", delimiter))
        }
    }
    for name in frame.names { titles.push(csv_field(name, delimiter)) }
    var lanes: list[list[str]] = []
    for level in gutter {
        lanes.push(rendered_column(level, na_rep, float_format, quote_for: delimiter))
    }
    for column in frame.columns {
        lanes.push(rendered_column(column, na_rep, float_format, quote_for: delimiter))
    }
    var pieces: list[str] = []
    if header and titles.len() > 0 {
        pieces.push(delimiter.join(titles))
        pieces.push("\n")
    }
    for row in 0..frame.len() {
        for (at, lane) in lanes.enumerate() {
            if at > 0 { pieces.push(delimiter) }
            pieces.push(lane[row])
        }
        pieces.push("\n")
    }
    return "".join(pieces)
}

#: Write a frame to `path` as delimited text. The arguments are
#: `to_csv_text`'s.
pub fn to_csv(
    frame: DataFrame,
    path: str,
    delimiter: str = ",",
    header: bool = true,
    index: bool = true,
    na_rep: str = "",
    float_format: str? = null
) -> void {
    sio.write_file(
        path,
        to_csv_text(
            frame,
            delimiter: delimiter,
            header: header,
            index: index,
            na_rep: na_rep,
            float_format: float_format
        )
    )
}

#: The five shapes `to_json` and `read_json` understand, as pandas names them.
pub let JSON_ORIENTS = ["records", "columns", "index", "split", "values"]

#: The member `to_json` writes its schema under, and `read_json` reads it back
#: from.
#:
#: One namespaced member rather than four loose ones: everything a reader needs
#: that pandas' `split` does not carry lives inside it, so another tool can
#: drop the member whole and still have a document it recognises.
pub let SCHEMA_MEMBER = "jaiframe"

fn check_orient(orient: str) -> void {
    for name in JSON_ORIENTS {
        if name == orient { return }
    }
    throw ValueError(f"orient {repr(orient)} is not one of {JSON_ORIENTS}")
}

#: Every row of a column as a JSON value, missing rows as `null`.
#:
#: `Datetime` goes out as its millisecond count rather than as ISO text: JSON
#: has no date, and an integer survives a reader that knows nothing about the
#: dtype list, where a string would come back as a string.
#:
#: The infinities go out as the strings JavaScript spells them with, because
#: RFC 8259 has no literal for them and `std.json` is strict. They come back as
#: floats in the orientations that record dtypes, and as text in the ones that
#: do not.
fn json_column(column: Column) -> list[any] {
    let length = column.length
    if length == 0 { return [] }
    var out: list[any] = []
    if column.dtype.is_dictionary() {
        let levels = column.levels ?? []
        for code in blanked(column.values, column.valid, length) {
            if is_nan(code) {
                out.push(null)
                continue
            }
            let at = int(code)
            out.push(if at >= 0 and at < levels.len() { levels[at] } else { null })
        }
        return out
    }
    if column.dtype.is_wide() {
        let (low, high) = blanked_wide(column)
        for (row, slot) in high.enumerate() {
            out.push(if is_nan(slot) { null } else { wide_unpack(low[row], slot) })
        }
        return out
    }
    let boolean = column.dtype == Dtype.Bool
    for value in blanked(column.values, column.valid, length) {
        if is_nan(value) {
            out.push(null)
            continue
        }
        if boolean {
            out.push(value != 0.0)
            continue
        }
        if value == INF {
            out.push("Infinity")
            continue
        }
        if value == -INF {
            out.push("-Infinity")
            continue
        }
        out.push(value)
    }
    return out
}

#: A JSON value back as the host value `column_from_values` wants.
fn from_json_cell(entry: any, dtype: Dtype?) -> any {
    if entry is null { return null }
    let target = dtype
    if target is null { return entry }
    if (target ?? Dtype.Float64) == Dtype.Datetime and isinstance(entry, int) {
        return Value.of_datetime(entry)
    }
    if isinstance(entry, str) {
        if entry == "Infinity" { return INF }
        if entry == "-Infinity" { return -INF }
    }
    return entry
}

#: Index labels as JSON: a flat list for one level, a list per row for several.
fn json_index(index: Index) -> list[any] {
    let levels = index.materialize()
    defer {
        for level in levels { level.free() }
    }
    let columns = levels.map(|level| json_column(level))
    if columns.len() == 1 { return columns[0] }
    var out: list[any] = []
    for row in 0..index.len() {
        var label: list[any] = []
        for column in columns { label.push(column[row]) }
        out.push(label)
    }
    return out
}

#: Index labels as the member names an object orientation needs.
fn json_label_names(index: Index) -> list[str] {
    if index.nlevels() > 1 {
        throw ValueError("a multi-index has no single member name; write it with orient \"split\"")
    }
    let levels = index.materialize()
    defer {
        for level in levels { level.free() }
    }
    return rendered_column(levels[0], "null", null)
}

#: What `read_json` needs that pandas' `split` does not carry.
fn schema_of(frame: DataFrame) -> dict[str, any] {
    var dtypes: list[any] = []
    for column in frame.columns { dtypes.push(column.dtype.name()) }
    var index_dtypes: list[any] = []
    if frame.index.is_range() {
        index_dtypes.push(Dtype.Int64.name())
    } else {
        for level in frame.index.levels { index_dtypes.push(level.dtype.name()) }
    }
    var index_names: list[any] = []
    for name in frame.index.names { index_names.push(name) }
    var schema: dict[str, any] = {
        "dtypes": dtypes,
        "index_dtypes": index_dtypes,
        "index_names": index_names,
        "index_range": null,
    }
    #: A range index says what its labels are in three numbers, and saying so
    #: is what lets the reader hand back the index the writer had rather than
    #: the stored ramp it stands for.
    if frame.index.is_range() {
        schema["index_range"] = [frame.index.start, frame.index.step]
    }
    return schema
}

#: Render a frame as JSON.
#:
#: The five orientations are pandas': `records` is a list of objects, `columns`
#: is `{column: {label: value}}`, `index` is `{label: {column: value}}`,
#: `values` is the bare data, and `split` keeps the three axes apart.
#:
#: `split` is the one to write when the file is going to be read back here. It
#: carries a `jaiframe` member holding the column dtypes, the index dtypes and
#: names, and a range index's start and step, and that is what makes the round
#: trip exact: `read_json(to_json(frame, orient: "split"), orient: "split")`
#: has the same dtypes, the same nulls, the same index and the same labels.
#: The other four orientations write values without types, so what comes back
#: is what inference makes of them — a `Datetime` column returns as `Int64`,
#: and an all-null column as `Float64`.
#:
#: `indent` at 0 is compact; a positive value puts every element on its own
#: line.
pub fn to_json_text(frame: DataFrame, orient: str = "columns", indent: int = 0) -> str {
    check_orient(orient)
    let data = frame.columns.map(|column| json_column(column))
    if orient == "values" {
        return sjson.stringify(json_rows(data, frame.len()), indent: indent)
    }
    if orient == "records" {
        var records: list[any] = []
        for row in 0..frame.len() {
            var record: dict[str, any] = {}
            for (at, name) in frame.names.enumerate() { record[name] = data[at][row] }
            records.push(record)
        }
        return sjson.stringify(records, indent: indent)
    }
    if orient == "split" {
        var document: dict[str, any] = {
            "index": json_index(frame.index),
            "columns": [name for name in frame.names],
            "data": json_rows(data, frame.len()),
        }
        document[SCHEMA_MEMBER] = schema_of(frame)
        return sjson.stringify(document, indent: indent)
    }
    let labels = json_label_names(frame.index)
    if orient == "columns" {
        var document: dict[str, any] = {}
        for (at, name) in frame.names.enumerate() {
            var cells: dict[str, any] = {}
            for (row, label) in labels.enumerate() { cells[label] = data[at][row] }
            document[name] = cells
        }
        return sjson.stringify(document, indent: indent)
    }
    var document: dict[str, any] = {}
    for (row, label) in labels.enumerate() {
        var cells: dict[str, any] = {}
        for (at, name) in frame.names.enumerate() { cells[name] = data[at][row] }
        document[label] = cells
    }
    return sjson.stringify(document, indent: indent)
}

fn json_rows(data: list[list[any]], rows: int) -> list[any] {
    var out: list[any] = []
    for row in 0..rows {
        var record: list[any] = []
        for column in data { record.push(column[row]) }
        out.push(record)
    }
    return out
}

#: Write a frame to `path` as JSON. The arguments are `to_json_text`'s.
pub fn to_json(frame: DataFrame, path: str, orient: str = "columns", indent: int = 0) -> void {
    sio.write_file(path, to_json_text(frame, orient: orient, indent: indent))
}

fn json_list(value: any, what: str) -> list[any] {
    if not isinstance(value, list) { throw ValueError(f"{what} must be a JSON array") }
    var out: list[any] = []
    for entry in value { out.push(entry) }
    return out
}

fn json_object(value: any, what: str) -> dict[str, any] {
    if not isinstance(value, dict) { throw ValueError(f"{what} must be a JSON object") }
    var out: dict[str, any] = {}
    for key in value.keys() { out[key] = value[key] }
    return out
}

#: Rows of JSON values turned the other way round, padded to `width`.
fn json_columns(rows: list[any], width: int) -> list[list[any]] {
    var lanes: list[list[any]] = []
    for _column in 0..width { lanes.push([]) }
    for (at, row) in rows.enumerate() {
        let cells = json_list(row, f"row {at}")
        if cells.len() > width {
            throw ValueError(f"row {at} has {cells.len()} values where the document names {width}")
        }
        for (position, lane) in lanes.enumerate() {
            lane.push(if position < cells.len() { cells[position] } else { null })
        }
    }
    return lanes
}

#: One column from JSON values, under the dtype the document recorded.
#:
#: The conversion from host values to lanes is `construct.jai`'s — it is the
#: same bulk upload every other constructor goes through — so all this does is
#: settle what each entry means before handing the lane over.
fn column_from_json(cells: list[any], dtype: Dtype?) -> Column {
    var values: list[any] = []
    for entry in cells { values.push(from_json_cell(entry, dtype)) }
    return column_from_values(values, dtype)
}

#: An index level from JSON labels, inferring the dtype when none was recorded.
fn level_from_json(labels: list[any], dtype: Dtype?) -> Column {
    return column_from_json(labels, dtype)
}

#: The member names of an object orientation, read as the labels they spell.
#:
#: JSON member names are text, so an index of integers writes as `"0"`, `"1"`
#: and would read back as an index of text. The same classification the CSV
#: reader uses puts them back — which is pandas' `convert_axes`, and the reason
#: a frame written with `orient: "index"` comes back with the index it had.
fn labels_from_names(names: list[str]) -> list[any] {
    var out: list[any] = []
    for name in names { out.push(literal_of(name)) }
    return out
}

#: Read JSON as a frame.
#:
#: `orient` must be the one the document was written with; nothing here guesses
#: at the shape, because `{"a": {"0": 1}}` is a legal `columns` document and a
#: legal `index` one and they do not mean the same table.
#:
#: A `split` document written by `to_json` carries its dtypes and comes back
#: exactly. The other four carry values only, so a column is whatever
#: inference makes of what it holds, and `dtypes` is how a caller says
#: otherwise.
pub fn read_json_text(
    text: str,
    orient: str = "columns",
    dtypes: dict[str, any]? = null
) -> DataFrame {
    check_orient(orient)
    let document = sjson.parse(text)
    let wanted = dtypes ?? {}
    var names: list[str] = []
    var lanes: list[list[any]] = []
    var index: Index? = null
    var recorded: list[Dtype?] = []
    if orient == "values" {
        let rows = json_list(document, "a values document")
        let width = if rows.len() == 0 { 0 } else { json_list(rows[0], "row 0").len() }
        names = numbered(width)
        lanes = json_columns(rows, width)
    } elif orient == "records" {
        let rows = json_list(document, "a records document")
        var seen: dict[str, bool] = {}
        var records: list[dict[str, any]] = []
        for (at, row) in rows.enumerate() {
            let record = json_object(row, f"record {at}")
            for name in record.keys() {
                if seen.get(name, false) { continue }
                seen[name] = true
                names.push(name)
            }
            records.push(record)
        }
        for name in names {
            var lane: list[any] = []
            for record in records { lane.push(record.get(name, null)) }
            lanes.push(lane)
        }
    } elif orient == "split" {
        let body = json_object(document, "a split document")
        names = json_list(body.get("columns", null), "the columns member").map(|name| f"{name}")
        lanes = json_columns(json_list(body.get("data", null), "the data member"), names.len())
        let schema = body.get(SCHEMA_MEMBER, null)
        if schema is not null {
            let described = json_object(schema, f"the {SCHEMA_MEMBER} member")
            for entry in json_list(described.get("dtypes", []), "the dtypes list") {
                recorded.push(as_dtype(entry))
            }
            index = index_from_schema(body.get("index", null), described, lanes)
        } else {
            index = index_from_labels(body.get("index", null), [], [])
        }
    } else {
        let body = json_object(document, f"a {orient} document")
        let outer = body.keys()
        var inner: list[str] = []
        var seen: dict[str, bool] = {}
        for key in outer {
            for name in json_object(body[key], f"member {key}").keys() {
                if seen.get(name, false) { continue }
                seen[name] = true
                inner.push(name)
            }
        }
        let (column_names, label_names) = if orient == "columns" {
            (outer, inner)
        } else {
            (inner, outer)
        }
        names = column_names
        for name in names {
            var lane: list[any] = []
            for label in label_names {
                let cells = if orient == "columns" {
                    json_object(body[name], f"member {name}")
                } else {
                    json_object(body[label], f"member {label}")
                }
                lane.push(cells.get(if orient == "columns" { label } else { name }, null))
            }
            lanes.push(lane)
        }
        if label_names.len() > 0 {
            let unnamed: list[str?] = [null]
            index = Index.of([column_from_json(labels_from_names(label_names), null)], unnamed)
        }
    }
    var columns: list[Column] = []
    for (at, name) in names.enumerate() {
        let override = wanted.get(name, null)
        var dtype: Dtype? = if at < recorded.len() { recorded[at] } else { null }
        if override is not null { dtype = as_dtype(override) }
        columns.push(column_from_json(lanes[at], dtype))
    }
    let rows = if lanes.len() > 0 { lanes[0].len() } else { index?.len() ?? 0 }
    return DataFrame(names, columns, index: index ?? Index.range(rows))
}

fn index_from_schema(labels: any, described: dict[str, any], lanes: list[list[any]]) -> Index? {
    let rows = if lanes.len() > 0 { lanes[0].len() } else { 0 }
    var names: list[str?] = []
    for entry in json_list(described.get("index_names", []), "the index_names list") {
        names.push(if entry is null { null } else { f"{entry}" })
    }
    let ramp = described.get("index_range", null)
    if ramp is not null {
        let pair = json_list(ramp, "the index_range pair")
        return Index.range(rows, start: int(f"{pair[0]}"), step: int(f"{pair[1]}"), name: if names.len() > 0 {
            names[0]
        } else {
            null
        })
    }
    var dtypes: list[Dtype?] = []
    for entry in json_list(described.get("index_dtypes", []), "the index_dtypes list") {
        dtypes.push(as_dtype(entry))
    }
    return index_from_labels(labels, dtypes, names)
}

fn index_from_labels(labels: any, dtypes: list[Dtype?], names: list[str?]) -> Index? {
    if labels is null { return null }
    let entries = json_list(labels, "the index member")
    if entries.len() == 0 { return null }
    let nested = isinstance(entries[0], list)
    var levels: list[list[any]] = []
    if nested {
        let width = json_list(entries[0], "index label 0").len()
        levels = json_columns(entries, width)
    } else {
        levels = [entries]
    }
    var columns: list[Column] = []
    for (at, level) in levels.enumerate() {
        columns.push(level_from_json(level, if at < dtypes.len() { dtypes[at] } else { null }))
    }
    var labelled: list[str?] = []
    for at in 0..columns.len() { labelled.push(if at < names.len() { names[at] } else { null }) }
    return Index.of(columns, labelled)
}

#: Read the JSON file at `path` as a frame. The arguments are
#: `read_json_text`'s.
pub fn read_json(path: str, orient: str = "columns", dtypes: dict[str, any]? = null) -> DataFrame {
    return read_json_text(sio.read_file(path), orient: orient, dtypes: dtypes)
}

#: What stands in for the rows and the columns a truncated view left out.
pub let ELLIPSIS = "..."

#: A laid-out table on its way to text: titles, cells, and which lanes are
#: numeric.
#:
#: Both renderings build one of these and neither builds the other's. The
#: cells are already strings and already windowed, so nothing downstream of
#: here knows how tall the frame was — which is what keeps printing a
#: ten-million-row frame the same cost as printing a sixty-row one.
class Sheet {
    #: One title per lane, the leading `gutter` of them naming index levels.
    pub var titles: list[str]
    #: One list of rendered cells per lane, all the same length.
    pub var lanes: list[list[str]]
    #: Whether a lane's cells are numbers, and so want the right margin.
    pub var numeric: list[bool]
    #: How many leading lanes are index levels rather than data.
    pub var gutter: int
    #: Rows and columns the frame has, which a truncated view still reports.
    pub var shape: (int, int)
    #: Whether rows or columns were left out.
    pub var truncated: bool

    pub fn init(self) {
        self.titles = []
        self.lanes = []
        self.numeric = []
        self.gutter = 0
        self.shape = (0, 0)
        self.truncated = false
    }

    pub fn height(self) -> int {
        return if self.lanes.len() == 0 { 0 } else { self.lanes[0].len() }
    }
}

#: How many rows to show at the top and at the bottom.
#:
#: A non-positive `max_rows` means all of them, which is what a file-bound
#: rendering like `to_markdown` wants and a terminal-bound one does not.
fn window(length: int, max_rows: int) -> (int, int) {
    if max_rows <= 0 or length <= max_rows { return (length, 0) }
    let head = (max_rows + 1) // 2
    return (head, max_rows - head)
}

#: A column's head and tail cells with an ellipsis between them.
fn windowed_cells(
    column: Column,
    head: int,
    tail: int,
    na_rep: str,
    float_format: str?
) -> list[str] {
    var cells = rendered_slice(column, 0, head, na_rep, float_format)
    if tail == 0 { return cells }
    cells.push(ELLIPSIS)
    for cell in rendered_slice(column, column.len() - tail, tail, na_rep, float_format) {
        cells.push(cell)
    }
    return cells
}

#: The labels of a range index, computed rather than read.
#:
#: The point of not storing the ramp is that most frames never ask for it, and
#: a printed frame is the case that would ask for ten million labels to show
#: sixty of them. Arithmetic on the two numbers it is made of costs nothing.
fn range_labels(index: Index, head: int, tail: int) -> list[str] {
    var out: list[str] = []
    for at in 0..head { out.push(f"{index.start + at * index.step}") }
    if tail == 0 { return out }
    out.push(ELLIPSIS)
    let first = index.length - tail
    for at in first..index.length { out.push(f"{index.start + at * index.step}") }
    return out
}

#: Put the index levels into a sheet as its leading lanes.
fn add_gutter(
    sheet: Sheet,
    index: Index,
    head: int,
    tail: int,
    na_rep: str,
    float_format: str?
) -> void {
    if index.is_range() {
        sheet.titles.push(index.names[0] ?? "")
        sheet.lanes.push(range_labels(index, head, tail))
        sheet.numeric.push(false)
        sheet.gutter = 1
        return
    }
    for (at, level) in index.levels.enumerate() {
        sheet.titles.push(index.names[at] ?? "")
        sheet.lanes.push(windowed_cells(level, head, tail, na_rep, float_format))
        sheet.numeric.push(false)
    }
    sheet.gutter = index.levels.len()
}

#: A frame laid out for printing.
#:
#: Columns are elided the way rows are: the leading and trailing ones survive
#: and an ellipsis lane stands where the middle was.
fn sheet_of_frame(
    frame: DataFrame,
    max_rows: int,
    max_cols: int,
    float_format: str?,
    na_rep: str,
    with_index: bool
) -> Sheet {
    let (head, tail) = window(frame.len(), max_rows)
    let (left, right) = window(frame.width(), max_cols)
    let sheet = Sheet()
    sheet.shape = frame.shape()
    sheet.truncated = tail > 0 or right > 0
    if with_index { add_gutter(sheet, frame.index, head, tail, na_rep, float_format) }
    let rows = head + tail + (if tail > 0 { 1 } else { 0 })
    for at in 0..left {
        sheet.titles.push(frame.names[at])
        sheet.lanes.push(windowed_cells(frame.columns[at], head, tail, na_rep, float_format))
        sheet.numeric.push(not frame.columns[at].dtype.is_dictionary())
    }
    if right > 0 {
        sheet.titles.push(ELLIPSIS)
        sheet.lanes.push([ELLIPSIS for _row in 0..rows])
        sheet.numeric.push(false)
        for at in frame.width() - right..frame.width() {
            sheet.titles.push(frame.names[at])
            sheet.lanes.push(windowed_cells(frame.columns[at], head, tail, na_rep, float_format))
            sheet.numeric.push(not frame.columns[at].dtype.is_dictionary())
        }
    }
    return sheet
}

#: One column laid out for printing, under whatever labels it has.
fn sheet_of_column(
    values: Column,
    index: Index,
    name: str?,
    max_rows: int,
    float_format: str?,
    na_rep: str,
    with_index: bool
) -> Sheet {
    let (head, tail) = window(values.len(), max_rows)
    let sheet = Sheet()
    sheet.shape = (values.len(), 1)
    sheet.truncated = tail > 0
    if with_index { add_gutter(sheet, index, head, tail, na_rep, float_format) }
    sheet.titles.push(name ?? "")
    sheet.lanes.push(windowed_cells(values, head, tail, na_rep, float_format))
    sheet.numeric.push(not values.dtype.is_dictionary())
    return sheet
}

fn justified(text: str, width: int, how: str) -> str {
    return match how {
        "left" => fmt.align_left(text, width),
        "right" => fmt.align_right(text, width),
        "center" => fmt.center(text, width),
        _ => throw ValueError(f"justify must be left, right or center, not {repr(how)}"),
    }
}

#: A sheet as a padded plain-text table.
fn draw_sheet(sheet: Sheet, justify: str, headed: bool) -> str {
    var widths: list[int] = []
    for (at, title) in sheet.titles.enumerate() {
        var width = if headed { fmt.display_width(title) } else { 0 }
        for cell in sheet.lanes[at] {
            let measured = fmt.display_width(cell)
            if measured > width { width = measured }
        }
        widths.push(width)
    }
    var lines: list[str] = []
    if headed {
        var titles: list[str] = []
        for (at, title) in sheet.titles.enumerate() {
            titles.push(justified(title, widths[at], if at < sheet.gutter {
                "left"
            } else {
                justify
            }))
        }
        lines.push("  ".join(titles).rstrip())
    }
    for row in 0..sheet.height() {
        var cells: list[str] = []
        for (at, lane) in sheet.lanes.enumerate() {
            #: The index gutter and text columns hang on the left margin, and
            #: numbers on the right, which is where a reader compares them.
            let how = if at < sheet.gutter or not sheet.numeric[at] { "left" } else { "right" }
            cells.push(justified(lane[row], widths[at], how))
        }
        lines.push("  ".join(cells).rstrip())
    }
    return "\n".join(lines)
}

#: A sheet as a GitHub-flavoured pipe table.
fn draw_markdown(sheet: Sheet) -> str {
    var widths: list[int] = []
    for (at, title) in sheet.titles.enumerate() {
        var width = fmt.display_width(title)
        if width < 3 { width = 3 }
        for cell in sheet.lanes[at] {
            let measured = fmt.display_width(cell)
            if measured > width { width = measured }
        }
        widths.push(width)
    }
    var lines: list[str] = []
    var titles: list[str] = []
    var rules: list[str] = []
    for (at, title) in sheet.titles.enumerate() {
        let right = at >= sheet.gutter and sheet.numeric[at]
        titles.push(justified(title, widths[at], if right { "right" } else { "left" }))
        let dashes = "-".repeat(widths[at] - 1)
        rules.push(if right { dashes + ":" } else { ":" + dashes })
    }
    lines.push("| " + " | ".join(titles) + " |")
    lines.push("| " + " | ".join(rules) + " |")
    for row in 0..sheet.height() {
        var cells: list[str] = []
        for (at, lane) in sheet.lanes.enumerate() {
            let right = at >= sheet.gutter and sheet.numeric[at]
            cells.push(justified(lane[row], widths[at], if right { "right" } else { "left" }))
        }
        lines.push("| " + " | ".join(cells) + " |")
    }
    return "\n".join(lines)
}

fn quoted_cells(cells: list[str], quote: bool) -> list[str] {
    if not quote { return cells }
    return cells.map(|cell| if cell == ELLIPSIS { cell } else { "\"" + cell + "\"" })
}

fn index_string(index: Index, max_rows: int, float_format: str?, na_rep: str) -> str {
    let named = index.names[0]
    if index.is_range() {
        let stop = index.start + index.length * index.step
        var out = f"RangeIndex(start={index.start}, stop={stop}, step={index.step}"
        if named is not null { out = out + f", name={repr(named ?? "")}" }
        return out + ")"
    }
    let (head, tail) = window(index.length, max_rows)
    var levels: list[list[str]] = []
    for level in index.levels {
        levels.push(
            quoted_cells(
                windowed_cells(level, head, tail, na_rep, float_format),
                level.dtype.is_dictionary()
            )
        )
    }
    if levels.len() == 1 {
        var out = f"Index([{", ".join(levels[0])}], dtype={repr(index.levels[0].dtype.name())}"
        if named is not null { out = out + f", name={repr(named ?? "")}" }
        return out + ")"
    }
    var labels: list[str] = []
    for row in 0..levels[0].len() {
        var parts: list[str] = []
        for level in levels { parts.push(level[row]) }
        labels.push("(" + ", ".join(parts) + ")")
    }
    var names: list[str] = []
    for name in index.names { names.push(if name is null { "null" } else { repr(name ?? "") }) }
    return f"MultiIndex([{", ".join(labels)}], names=[{", ".join(names)}])"
}

fn frame_string(
    frame: DataFrame,
    max_rows: int,
    max_cols: int,
    float_format: str?,
    justify: str,
    na_rep: str
) -> str {
    if frame.width() == 0 or frame.len() == 0 {
        var lines = ["Empty DataFrame", "Columns: [" + ", ".join(frame.names) + "]"]
        lines.push("Index: " + index_string(frame.index, max_rows, float_format, na_rep))
        return "\n".join(lines)
    }
    let sheet = sheet_of_frame(frame, max_rows, max_cols, float_format, na_rep, true)
    let body = draw_sheet(sheet, justify, true)
    if not sheet.truncated { return body }
    let (rows, columns) = sheet.shape
    return body + f"\n\n[{rows} rows x {columns} columns]"
}

fn column_string(
    values: Column,
    index: Index,
    name: str?,
    max_rows: int,
    float_format: str?,
    justify: str,
    na_rep: str
) -> str {
    var footer: list[str] = []
    if name is not null { footer.push(f"Name: {name ?? ""}") }
    if values.len() == 0 {
        footer.push("Empty")
    }
    let sheet = sheet_of_column(values, index, name, max_rows, float_format, na_rep, true)
    if sheet.truncated { footer.push(f"Length: {values.len()}") }
    footer.push(f"dtype: {values.dtype.name()}")
    let tail = ", ".join(footer)
    if values.len() == 0 { return tail }
    #: A series has no header row: its name belongs in the footer, where the
    #: dtype is, and a header would put it somewhere a reader has to look
    #: twice to attach it to anything.
    return draw_sheet(sheet, justify, false) + "\n" + tail
}

#: A frame, series, column or index as printable text.
#:
#: pandas' truncation: past `max_rows` the head and the tail survive with an
#: ellipsis row between them, and past `max_cols` the leading and trailing
#: columns do with an ellipsis column. A non-positive limit means no limit.
#: Numbers sit against the right margin and text against the left, which is
#: where each is read from; `justify` moves the column titles and takes
#: `"left"`, `"right"` or `"center"`.
#:
#: `float_format` is a `str.format` spec, and without one a float prints as
#: the shortest decimal that reads back as the same float32. A missing cell
#: prints as `na_rep`.
#:
#: Only the rows it shows are read: a truncated view downloads the head slice
#: and the tail slice of each lane and nothing between them.
pub fn to_string(
    target: any,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
    float_format: str? = null,
    justify: str = "right",
    na_rep: str = "NaN"
) -> str {
    if isinstance(target, DataFrame) {
        return frame_string(target, max_rows, max_cols, float_format, justify, na_rep)
    }
    if isinstance(target, Series) {
        return column_string(
            target.values,
            target.index,
            target.name,
            max_rows,
            float_format,
            justify,
            na_rep
        )
    }
    if isinstance(target, Column) {
        return column_string(
            target,
            Index.range(target.len()),
            null,
            max_rows,
            float_format,
            justify,
            na_rep
        )
    }
    if isinstance(target, Index) { return index_string(target, max_rows, float_format, na_rep) }
    throw TypeError(f"{repr(target)} is not a frame, a series, a column or an index")
}

#: A frame, series or column as a GitHub-flavoured markdown table.
#:
#: `max_rows` defaults to no limit: a markdown table goes into a document
#: rather than onto a screen, and a document that silently dropped the middle
#: of its data would be a worse document. The alignment colons follow the
#: dtypes, so a numeric column right-aligns wherever the table is rendered.
pub fn to_markdown(
    target: any,
    index: bool = true,
    max_rows: int = 0,
    float_format: str? = null,
    na_rep: str = ""
) -> str {
    if isinstance(target, DataFrame) {
        return draw_markdown(sheet_of_frame(target, max_rows, 0, float_format, na_rep, index))
    }
    if isinstance(target, Series) {
        return draw_markdown(
            sheet_of_column(
                target.values,
                target.index,
                target.name,
                max_rows,
                float_format,
                na_rep,
                index
            )
        )
    }
    if isinstance(target, Column) {
        return draw_markdown(
            sheet_of_column(
                target,
                Index.range(target.len()),
                null,
                max_rows,
                float_format,
                na_rep,
                index
            )
        )
    }
    throw TypeError(f"{repr(target)} is not a frame, a series or a column")
}

#: How many float32 slots a column occupies, over all the lanes it has.
#:
#: This is the honest unit for this package: a column is one contiguous buffer
#: per lane and nothing else, so its footprint is a slot count and everything
#: else is arithmetic on it. `Int64` and `Datetime` pay a second lane for their
#: exactness; a column that has never held a null pays for no mask.
pub fn buffer_slots(column: Column) -> int {
    var lanes = 1
    if column.dtype.is_wide() { lanes += 1 }
    if column.has_validity() { lanes += 1 }
    return column.len() * lanes
}

#: How many slots an index occupies. A range index occupies none: its labels
#: are two integers and the arithmetic between them.
pub fn index_slots(index: Index) -> int {
    var total = 0
    for level in index.levels { total += buffer_slots(level) }
    return total
}

#: The bytes the strings of a dictionary column occupy, counted once per row.
#:
#: What a `Str` column would cost if it were not dictionary-encoded, which is
#: the number `deep` is asking for. Reduced on the device — one gather out of
#: the level table per row and one fold — so a ten-million-row column costs a
#: dispatch and a download of a few hundred partials, not ten million host
#: string lengths.
fn text_bytes(column: Column) -> int {
    let levels = column.levels ?? []
    if column.length == 0 or levels.len() == 0 { return 0 }
    let lengths = Buffer(levels.len())
    lengths.upload(levels.map(|level| float(bytes(level).len())))
    let low = Buffer(BYTES_GROUPS)
    let high = Buffer(BYTES_GROUPS)
    defer {
        lengths.free()
        low.free()
        high.free()
    }
    compiled("io_text_bytes").dispatch_async(
        [column.values, column.valid ?? column.values, lengths, low, high],
        BYTES_GROUPS * BYTES_GROUP,
        [column.length, levels.len(), flag(column.has_validity())],
        group_size: BYTES_GROUP
    )
    let lows = low.download()
    let highs = high.download()
    var total = 0
    for (at, slot) in lows.enumerate() { total += wide_unpack(slot, highs[at]) }
    return total
}

#: How many bytes a column occupies.
pub fn column_bytes(column: Column, deep: bool = false) -> int {
    var total = buffer_slots(column) * 4
    if deep and column.dtype.is_dictionary() { total += text_bytes(column) }
    return total
}

#: What each column of a frame costs, in bytes, as a series over the names.
#:
#: A slot is four bytes, so every number here is `buffer_slots` times four; the
#: slot count is the thing the package actually allocates and the byte count is
#: the thing a reader is asking about. `index` prepends the index's own cost,
#: which for the default range index is zero.
#:
#: `deep` adds what a dictionary column's strings occupy — the size the column
#: would have been without the encoding — counted on the device rather than by
#: walking the rows.
pub fn memory_usage(frame: DataFrame, index: bool = true, deep: bool = false) -> Series {
    var names: list[str?] = []
    var counts: list[int] = []
    if index {
        names.push("Index")
        counts.push(index_slots(frame.index) * 4)
    }
    for (at, column) in frame.columns.enumerate() {
        names.push(frame.names[at])
        counts.push(column_bytes(column, deep))
    }
    let labels: list[str?] = [null]
    return Series(
        Column.from_ints(counts),
        index: Index.of([Column.from_strs(names)], labels),
        name: "memory_usage"
    )
}

#: A summary of a frame: its shape, its index, and every column's dtype and
#: how many of its rows are present.
#:
#: The non-null counts are the device's answer — one reduction per column
#: through `Column.null_count` — so this costs a dispatch a column rather than
#: a download of the whole table.
pub fn info(frame: DataFrame, deep: bool = false) -> str {
    var lines = [f"<DataFrame: {frame.len()} rows x {frame.width()} columns>"]
    lines.push("Index: " + to_string(frame.index, max_rows: 6))
    var rows: list[list[str]] = [["#", "Column", "Non-Null Count", "Dtype"]]
    var tally: dict[str, int] = {}
    for (at, column) in frame.columns.enumerate() {
        let present = column.len() - column.null_count()
        rows.push([f"{at}", frame.names[at], f"{present} non-null", column.dtype.name()])
        let name = column.dtype.name()
        tally[name] = tally.get(name, 0) + 1
    }
    var widths = [0, 0, 0, 0]
    for row in rows {
        for (at, cell) in row.enumerate() {
            let measured = fmt.display_width(cell)
            if measured > widths[at] { widths[at] = measured }
        }
    }
    for row in rows {
        var padded: list[str] = []
        for (at, cell) in row.enumerate() { padded.push(fmt.align_left(cell, widths[at])) }
        lines.push(" ".join(padded).rstrip())
    }
    var kinds = tally.keys()
    kinds.sort()
    var counted: list[str] = []
    for kind in kinds { counted.push(f"{kind}({tally.get(kind, 0)})") }
    lines.push("dtypes: " + ", ".join(counted))
    var slots = index_slots(frame.index)
    var total = slots * 4
    for column in frame.columns {
        slots += buffer_slots(column)
        total += column_bytes(column, deep)
    }
    lines.push(f"memory usage: {total} bytes over {slots} slots")
    return "\n".join(lines)
}
