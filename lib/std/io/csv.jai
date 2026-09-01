#: Delimited text: RFC 4180 comma-separated values and the tab-separated form.
from .stream import read_file, write_file

#: Parse `text` as delimited values, following RFC 4180 quoting.
#:
#: A field may be wrapped in double quotes, inside which a doubled quote is one
#: literal quote and the delimiter and newlines are ordinary characters.
#: Records end at `\n` or `\r\n`, and a terminator at the end of the input does
#: not produce an extra record. A blank line is one record holding one empty
#: field. An unterminated quoted field raises `ParseError`.
#:
#: ```
#: parse_csv("a,b\n1,2\n")                 # [["a", "b"], ["1", "2"]]
#: parse_csv("\"x,y\",\"he said \"\"hi\"\"\"")
#: # [["x,y", "he said \"hi\""]]
#: ```
pub fn parse_csv(text: str, delimiter: str = ",") -> list[list[str]] {
    _check_delimiter(delimiter)
    var rows: list[list[str]] = []
    var row: list[str] = []
    var field: list[str] = []
    var quoted = false
    var started = false
    var index = 0
    while index < text.len() {
        let character = text[index]
        index += 1
        if quoted {
            if character != "\"" {
                field.push(character)
                continue
            }
            if index < text.len() and text[index] == "\"" {
                field.push("\"")
                index += 1
                continue
            }
            quoted = false
            continue
        }
        if character == "\"" and field.len() == 0 {
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
            if character == "\r" and index < text.len() and text[index] == "\n" { index += 1 }
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

#: Render `rows` as delimited values, quoting the fields that need it.
#:
#: A field is quoted when it contains the delimiter, a double quote, a newline
#: or a carriage return, or when it begins or ends with a space — the cases
#: where an unquoted field would not read back as itself. Every record,
#: including the last, ends with `\n`.
#:
#: ```
#: format_csv([["a", "b"], ["1", "x,y"]])    # "a,b\n1,\"x,y\"\n"
#: format_csv([])                            # ""
#: ```
pub fn format_csv(rows: list[list[str]], delimiter: str = ",") -> str {
    _check_delimiter(delimiter)
    var out: list[str] = []
    for row in rows {
        out.push(delimiter.join([_quote_field(cell, delimiter) for cell in row]))
        out.push("\n")
    }
    return "".join(out)
}

#: Read the file at `path` as delimited values.
pub fn read_csv(path: str, delimiter: str = ",") -> list[list[str]] {
    return parse_csv(read_file(path), delimiter)
}

#: Write `rows` to `path` as delimited values.
pub fn write_csv(path: str, rows: list[list[str]], delimiter: str = ",") -> void {
    write_file(path, format_csv(rows, delimiter))
}

#: Read the file at `path` as tab-separated values.
#:
#: Quoting works as it does for CSV, which is what spreadsheets emit for TSV
#: and the only way a field containing a tab can survive the round trip.
pub fn read_tsv(path: str) -> list[list[str]] { return read_csv(path, "\t") }

#: Write `rows` to `path` as tab-separated values.
pub fn write_tsv(path: str, rows: list[list[str]]) -> void { write_csv(path, rows, "\t") }

fn _check_delimiter(delimiter: str) -> void {
    if delimiter.len() != 1 {
        throw ValueError(f"delimiter must be one character, got {repr(delimiter)}")
    }
    if delimiter == "\"" or delimiter == "\n" or delimiter == "\r" {
        throw ValueError(f"delimiter may not be a quote or a newline, got {repr(delimiter)}")
    }
}

fn _quote_field(cell: str, delimiter: str) -> str {
    if not _needs_quoting(cell, delimiter) { return cell }
    var parts = ["\""]
    for index in 0..cell.len() {
        parts.push(cell[index] == "\"" ? "\"\"" : cell[index])
    }
    parts.push("\"")
    return "".join(parts)
}

fn _needs_quoting(cell: str, delimiter: str) -> bool {
    if cell.len() == 0 { return false }
    if cell[0] == " " or cell[cell.len() - 1] == " " { return true }
    for index in 0..cell.len() {
        let character = cell[index]
        if character == delimiter or character == "\"" { return true }
        if character == "\n" or character == "\r" { return true }
    }
    return false
}
