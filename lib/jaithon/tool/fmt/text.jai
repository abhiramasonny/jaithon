#: Text the printer, the scanner and the Jaithon 2 rewrite all walk by hand.
pub const _INDENT: str = "    "

#: How many triple-quote marks a line opens or closes. Odd means the line
#: changes which side of a multi-line string the next line is on.
pub fn _triple_quotes(piece: str) -> int {
    var found = 0
    var at = 0
    while at + 3 <= piece.len() {
        if piece[at:at + 3] == "\"\"\"" {
            found += 1
            at += 3
        } else {
            at += 1
        }
    }
    return found
}

pub fn _indent_block(text: str) -> str {
    return _INDENT + text.replace("\n", "\n" + _INDENT)
}

pub fn _strip_end(text: str) -> str {
    var end = text.len()
    while end > 0 and (text[end - 1] == " " or text[end - 1] == "\t") { end -= 1 }
    return text[0:end]
}

pub fn _leading_space(text: str) -> str {
    var index = 0
    while index < text.len() and (text[index] == " " or text[index] == "\t") { index += 1 }
    return text[0:index]
}

pub fn _lines_of(text: str) -> list[str] {
    let lines = text.split("\n")
    if lines.len() > 0 and lines[lines.len() - 1].len() == 0 { lines.pop() }
    return lines
}

pub fn _dedent_continuations(text: str, column: int) -> str {
    if not text.contains("\n") { return text }
    let lines = text.split("\n")
    let out: list[str] = [lines[0]]
    for index in 1..lines.len() {
        let line = lines[index]
        let cut = min(column - 1, _leading_space(line).len())
        out.push(line[cut:line.len()])
    }
    return "\n".join(out)
}

pub fn _utf8_length(code_point: int) -> int {
    if code_point < 0x80 { return 1 }
    if code_point < 0x800 { return 2 }
    if code_point < 0x10000 { return 3 }
    return 4
}

pub fn _skip_string(
    chars: list[str],
    start: int,
    line: int,
    column: int,
    raw: bool
) -> tuple[int, int, int] {
    let length = chars.len()
    let quote = chars[start]
    var index = start + 1
    var current_line = line
    var current_column = column + 1
    var triple = false
    if index + 1 < length and chars[index] == quote and chars[index + 1] == quote {
        triple = true
        index += 2
        current_column += 2
    }
    while index < length {
        let character = chars[index]
        if character == "\\" and not raw {
            index += 2
            current_column += 2
        } elif character == "\n" {
            current_line += 1
            current_column = 1
            index += 1
        } elif character == quote and not triple {
            return (index + 1, current_line, current_column + 1)
        } elif character == quote and
            triple and
            index + 2 < length and
            chars[index + 1] == quote and
            chars[index + 2] == quote {
            return (index + 3, current_line, current_column + 3)
        } else {
            index += 1
            current_column += 1
        }
    }
    return (length, current_line, current_column)
}
