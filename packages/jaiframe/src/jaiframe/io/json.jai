#: JSON in and out in the five orientations pandas names, with the schema
#: member that makes `split` an exact round trip.
import std.io as sio
import std.json as sjson
from std.math import INF, is_nan

from ..column import Column, DataFrame, Dtype, Index, Value
from ..column import wide_unpack
from ..construct import column_from_values
from .csv import as_dtype, literal_of, numbered
from .kernels import blanked, blanked_wide
from .render import rendered_column

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
