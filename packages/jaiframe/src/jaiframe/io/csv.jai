#: Delimited text in and out: the record split, the ladder a field climbs to
#: settle its dtype, and `read_csv` and `to_csv` themselves.
import std.io as sio
import std.str as text_ops
from std.math import NAN

from ..column import Column, DataFrame, Dtype, Index
from ..construct import infer_dtype
from .render import csv_field, rendered_column
from ..timeseries import to_datetime

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
pub fn literal_of(field: str) -> any {
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

pub fn numbered(width: int) -> list[str] {
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

pub fn as_dtype(spec: any) -> Dtype {
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
