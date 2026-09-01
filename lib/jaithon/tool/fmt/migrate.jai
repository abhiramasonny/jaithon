#: The mechanical Jaithon 2 to Jaithon 3 rewrite behind `--migrate2`.
from std.str import StringBuilder
from .text import _leading_space, _lines_of, _skip_string

#: Rewrite Jaithon 2 source into Jaithon 3, mechanically.
#:
#: The table in `spec/LANGUAGE.md` §16 applied to text (Jaithon 2 source cannot
#: be parsed into a Jaithon 3 tree). What cannot be done mechanically — a
#: `namespace` block — is marked with a `# TODO(migrate2):` comment instead of
#: guessed at. The result is not guaranteed to compile.
#:
#: ```
#: migrate2("print x\n")              # "print(x)\n"
#: migrate2("var n = 2 ^ 3\n")        # "var n = 2 ** 3\n"
#: migrate2("if a eq b then do\n")    # "if a == b {\n"
#: migrate2("end\n")                  # "}\n"
#: ```
pub fn migrate2(source: str) -> str {
    let out: list[str] = []
    for raw in _lines_of(source) {
        let indent = _leading_space(raw)
        let body = raw.strip()
        if body.len() == 0 {
            out.push("")
        } elif body.starts_with("#") {
            out.push(raw)
        } elif body == "end" {
            out.push(indent + "}")
        } else {
            out.push(indent + _migrate_line(body))
        }
    }
    return "\n".join(out) + "\n"
}

let _MIGRATE_WORDS: dict[str, str] = {"eq": "==", "func": "fn", "elseif": "elif"}

fn _migrate_line(body: str) -> str {
    if body.starts_with("namespace ") {
        return "# TODO(migrate2): `" +
            body +
            "` becomes a module file or a class of statics (spec §16)"
    }
    if body.contains("static void main") { return "pub fn main() -> int {" }
    if body == "print" { return "print()" }
    if body.starts_with("print ") {
        return "print(" + _rewrite_code(body[6:body.len()].strip()) + ")"
    }
    if body.starts_with("input ") {
        let name = body[6:body.len()].strip()
        return f"let {name} = input()"
    }
    if body.starts_with("system ") {
        return "os.shell(" + body[7:body.len()].strip() + ")"
    }
    if body.starts_with("import ") and body.contains("/") {
        return "import std." + body[7:body.len()].strip().split("/")[0]
    }
    return _migrate_block_words(_rewrite_code(body))
}

fn _migrate_block_words(text: str) -> str {
    var out = text
    if out.ends_with(" then do") {
        out = out[0:out.len() - 8] + " {"
    } elif out.ends_with(" then") {
        out = out[0:out.len() - 5] + " {"
    } elif out.ends_with(" do") {
        out = out[0:out.len() - 3] + " {"
    }
    if out == "else" { return "} else {" }
    if out.starts_with("else ") { return "} else " + out[5:out.len()] }
    if out.starts_with("fn ") and not out.ends_with("{") { return out + " {" }
    return out
}

fn _rewrite_code(text: str) -> str {
    let out = StringBuilder()
    let chars = [character for character in text]
    var index = 0
    while index < chars.len() {
        let character = chars[index]
        if character == "\"" or character == "'" {
            let raw = index > 0 and chars[index - 1] == "r"
            let (next, _line, _column) = _skip_string(chars, index, 1, 1, raw)
            out.push("".join(chars[index:next]))
            index = next
        } elif character == "^" {
            out.push("**")
            index += 1
        } elif character == "!" and (index + 1 >= chars.len() or chars[index + 1] != "=") {
            let (head, operand) = _split_factorial_operand(out.build())
            out.clear()
            out.push(head + "math.factorial(" + operand + ")")
            index += 1
        } elif _is_ident_start(character) {
            index = _rewrite_word(out, chars, index)
        } else {
            out.push(character)
            index += 1
        }
    }
    return out.build()
}

fn _rewrite_word(out: StringBuilder, chars: list[str], index: int) -> int {
    var end = index
    while end < chars.len() and _is_ident_part(chars[end]) { end += 1 }
    let word = "".join(chars[index:end])

    if word == "new" {
        var next = end
        while next < chars.len() and chars[next] == " " { next += 1 }
        return next
    }
    if word == "_alen" or word == "arrayLen" {
        let (next, argument) = _call_argument(chars, end)
        if argument.len() > 0 {
            out.push(argument.strip() + ".len()")
            return next
        }
    }
    if word == "_get" or word == "arrayGet" {
        let (next, argument) = _call_argument(chars, end)
        let parts = _split_top_level(argument)
        if parts.len() == 2 {
            out.push(parts[0].strip() + "[" + parts[1].strip() + "]")
            return next
        }
    }
    out.push(_MIGRATE_WORDS.get(word, word))
    return end
}

fn _split_factorial_operand(text: str) -> tuple[str, str] {
    let end = text.len()
    if end == 0 { return ("", "0") }
    var index = end
    if text[end - 1] == ")" {
        var depth = 0
        index = end - 1
        while index >= 0 {
            if text[index] == ")" { depth += 1 }
            if text[index] == "(" {
                depth -= 1
                if depth == 0 { break }
            }
            index -= 1
        }
    }
    while index > 0 and (_is_ident_part(text[index - 1]) or text[index - 1] == ".") {
        index -= 1
    }
    return (text[0:index], text[index:end])
}

fn _call_argument(chars: list[str], open: int) -> tuple[int, str] {
    var index = open
    while index < chars.len() and chars[index] == " " { index += 1 }
    if index >= chars.len() or chars[index] != "(" { return (open, "") }
    var depth = 0
    let start = index + 1
    while index < chars.len() {
        if chars[index] == "(" { depth += 1 }
        if chars[index] == ")" {
            depth -= 1
            if depth == 0 { break }
        }
        index += 1
    }
    return (index + 1, "".join(chars[start:index]))
}

fn _split_top_level(text: str) -> list[str] {
    let parts: list[str] = []
    var depth = 0
    var start = 0
    for index in 0..text.len() {
        let character = text[index]
        if character == "(" or character == "[" { depth += 1 }
        if character == ")" or character == "]" { depth -= 1 }
        if character == "," and depth == 0 {
            parts.push(text[start:index])
            start = index + 1
        }
    }
    parts.push(text[start:text.len()])
    return parts
}

pub fn _is_ident_start(character: str) -> bool {
    return character == "_" or character.lower() != character.upper()
}

pub fn _is_ident_part(character: str) -> bool {
    return _is_ident_start(character) or character >= "0" and character <= "9"
}
