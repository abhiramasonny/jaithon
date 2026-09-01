#: The prototxt half: protobuf's text format, read a character at a time.
from std.math import INF, NAN
from .message import Message
from .schema import ENUMS, MAX_DEPTH, Spec, layout_of

fn is_digit(character: str) -> bool { return character >= "0" and character <= "9" }

fn is_octal(character: str) -> bool { return character >= "0" and character <= "7" }

fn is_letter(character: str) -> bool {
    return character >= "a" and character <= "z" or character >= "A" and character <= "Z"
}

fn hex_value(character: str) -> int {
    if is_digit(character) { return ord(character) - ord("0") }
    if character >= "a" and character <= "f" { return ord(character) - ord("a") + 10 }
    if character >= "A" and character <= "F" { return ord(character) - ord("A") + 10 }
    return -1
}

#: The one-character escapes protobuf's text format defines, by the character
#: that introduces them.
let ESCAPES: dict[str, int] = {
    "a": 7,
    "b": 8,
    "f": 12,
    "n": 10,
    "r": 13,
    "t": 9,
    "v": 11,
    "?": 63,
    "'": 39,
    "\"": 34,
    "\\": 92,
}

#: A prototxt, read a character at a time.
#:
#: The whole file is a `str` rather than `bytes` because indexing one is
#: constant time here and the syntax is ASCII; only the inside of a quoted
#: string is assembled as bytes, since an escape there names a byte and not a
#: character.
class TextReader {
    pub let text: str
    pub let size: int
    pub var at: int

    pub fn init(self, text: str) {
        self.text = text
        self.size = text.len()
        self.at = 0
    }

    #: Where the cursor is, for an error. Counting from the start each time is
    #: fine because this only ever runs on the way to a throw.
    pub fn place(self) -> str {
        var line = 1
        for index in 0..self.at {
            if self.text[index] == "\n" { line += 1 }
        }
        return f"line {line} of this prototxt"
    }

    pub fn peek(self) -> str {
        if self.at >= self.size { return "" }
        return self.text[self.at]
    }

    #: Whitespace, and `#` comments, which run to the end of the line.
    pub fn skip_blank(self) -> void {
        while self.at < self.size {
            let character = self.text[self.at]
            if character == "#" {
                while self.at < self.size and self.text[self.at] != "\n" { self.at += 1 }
            } elif character == " " or
                character == "\t" or
                character == "\n" or
                character == "\r" or
                character == "\x0b" or
                character == "\x0c" {
                self.at += 1
            } else {
                return
            }
        }
    }

    pub fn read_name(self) -> str {
        let start = self.at
        while self.at < self.size {
            let character = self.text[self.at]
            if is_letter(character) or is_digit(character) or character == "_" {
                self.at += 1
            } else {
                break
            }
        }
        if self.at == start {
            throw ValueError(f"{self.place()}: expected a field name, found {self.peek()}")
        }
        return self.text[start:self.at]
    }

    #: Everything up to the next delimiter: one number, enum identifier or
    #: boolean.
    pub fn read_token(self) -> str {
        let start = self.at
        while self.at < self.size {
            let character = self.text[self.at]
            if character == " " or
                character == "\t" or
                character == "\n" or
                character == "\r" or
                character == "," or
                character == ";" or
                character == ":" or
                character == "{" or
                character == "}" or
                character == "<" or
                character == ">" or
                character == "[" or
                character == "]" or
                character == "#" {
                break
            }
            self.at += 1
        }
        if self.at == start {
            throw ValueError(f"{self.place()}: expected a value, found {self.peek()}")
        }
        return self.text[start:self.at]
    }

    #: One quoted literal. Escapes name bytes, so the content is assembled as
    #: bytes and decoded once at the end rather than a character at a time --
    #: `\xc3\xa9` is one character and not two.
    pub fn read_quoted(self) -> str {
        let quote = self.peek()
        self.at += 1
        var raw: list[int] = []
        loop {
            if self.at >= self.size {
                throw ValueError(f"{self.place()}: this string never closes")
            }
            let character = self.text[self.at]
            if character == quote {
                self.at += 1
                break
            }
            if character == "\n" {
                throw ValueError(f"{self.place()}: this string runs past the end of its line")
            }
            if character != "\\" {
                for byte in character.bytes() { raw.push(byte) }
                self.at += 1
                continue
            }
            self.at += 1
            if self.at >= self.size {
                throw ValueError(f"{self.place()}: this string ends on a backslash")
            }
            let marker = self.text[self.at]
            self.at += 1
            let simple = ESCAPES.get(marker, -1)
            if simple >= 0 {
                raw.push(simple)
            } elif marker == "x" or marker == "X" {
                var value = -1
                for _digit in 0..2 {
                    let digit = hex_value(self.peek())
                    if digit < 0 { break }
                    value = if value < 0 { digit } else { value * 16 + digit }
                    self.at += 1
                }
                if value < 0 {
                    throw ValueError(f"{self.place()}: \\{marker} needs at least one hex digit")
                }
                raw.push(value)
            } elif is_octal(marker) {
                var value = ord(marker) - ord("0")
                for _digit in 0..2 {
                    if not is_octal(self.peek()) { break }
                    value = value * 8 + ord(self.peek()) - ord("0")
                    self.at += 1
                }
                if value > 255 {
                    throw ValueError(f"{self.place()}: the octal escape \\{value} is above a byte")
                }
                raw.push(value)
            } elif marker == "u" or marker == "U" {
                let width = if marker == "u" { 4 } else { 8 }
                var value = 0
                for _digit in 0..width {
                    let digit = hex_value(self.peek())
                    if digit < 0 {
                        throw ValueError(
                            f"{self.place()}: \\{marker} needs exactly {width} hex digits"
                        )
                    }
                    value = value * 16 + digit
                    self.at += 1
                }
                for byte in chr(value).bytes() { raw.push(byte) }
            } else {
                throw ValueError(f"{self.place()}: \\{marker} is not an escape")
            }
        }
        return __prim__.str_decode(bytes(raw))
    }

    #: Adjacent literals, which the text format joins into one string.
    pub fn read_string(self) -> str {
        var joined = self.read_quoted()
        loop {
            let mark = self.at
            self.skip_blank()
            if self.peek() != "\"" and self.peek() != "'" {
                self.at = mark
                return joined
            }
            joined = joined + self.read_quoted()
        }
    }
}

#: A whole number the way protobuf's text format writes one: decimal, `0x`
#: hex, or a leading zero for octal.
fn text_int(token: str, place: str, name: str) -> int {
    var body = token
    var sign = 1
    if body.starts_with("-") {
        sign = -1
        body = body[1:]
    } elif body.starts_with("+") {
        body = body[1:]
    }
    var base = 10
    if body.starts_with("0x") or body.starts_with("0X") {
        base = 16
        body = body[2:]
    } elif body.len() > 1 and body.starts_with("0") {
        base = 8
        body = body[1:]
    }
    if body.len() == 0 { throw ValueError(f"{place}: {name} has no digits in {token}") }
    var value = 0
    for index in 0..body.len() {
        let digit = hex_value(body[index])
        if digit < 0 or digit >= base {
            throw ValueError(f"{place}: {name} is not a whole number in {token}")
        }
        value = value * base + digit
    }
    return sign * value
}

#: A float the way the text format writes one, including the `f` suffix a
#: hand-written prototxt often carries and the three spelled-out values.
fn text_float(token: str, place: str, name: str) -> float {
    var body = token
    var sign = 1.0
    if body.starts_with("-") {
        sign = -1.0
        body = body[1:]
    } elif body.starts_with("+") {
        body = body[1:]
    }
    let folded = body.lower()
    if folded == "inf" or folded == "infinity" { return sign * INF }
    if folded == "nan" { return NAN }
    if body.ends_with("f") or body.ends_with("F") { body = body[:body.len() - 1] }
    if body.starts_with("0x") or body.starts_with("0X") {
        return sign * float(text_int(body, place, name))
    }
    let parsed = body.to_float()
    if parsed is null { throw ValueError(f"{place}: {name} is not a number in {token}") }
    let value: float = parsed
    return sign * value
}

fn text_bool(token: str, place: str, name: str) -> int {
    if token == "true" or token == "True" or token == "t" or token == "1" { return 1 }
    if token == "false" or token == "False" or token == "f" or token == "0" { return 0 }
    throw ValueError(f"{place}: {name} is a boolean and {token} is not one")
}

fn text_enum(token: str, spec: Spec, place: str) -> int {
    let table = ENUMS.get(spec.of, null)
    if table is not null {
        let found = table.get(token, null)
        if found is not null { return found }
    }
    if token.len() > 0 and (is_digit(token[0]) or token[0] == "-" or token[0] == "+") {
        return text_int(token, place, spec.name)
    }
    throw ValueError(f"{place}: {token} is not one of {spec.of}'s values")
}

#: Everything between an opening brace and its match, thrown away. This runs
#: for a field the schema does not model, which is most of what a real
#: prototxt holds, so it works off the syntax alone.
fn skip_body(reader: TextReader, shut: str, depth: int) -> void {
    if depth > MAX_DEPTH {
        throw ValueError(f"{reader.place()}: this prototxt nests more than {MAX_DEPTH} deep")
    }
    loop {
        reader.skip_blank()
        if reader.at >= reader.size {
            throw ValueError(f"this prototxt ends with no closing {shut}")
        }
        let character = reader.peek()
        if character == shut {
            reader.at += 1
            return
        }
        if character == "\"" or character == "'" {
            let _text = reader.read_quoted()
            continue
        }
        if character == "{" or character == "<" {
            reader.at += 1
            skip_body(reader, if character == "{" { "}" } else { ">" }, depth + 1)
            continue
        }
        if character == "}" or character == ">" {
            throw ValueError(f"{reader.place()}: found {character} where {shut} was expected")
        }
        reader.at += 1
    }
}

#: One scalar value, stored if the schema knows the field and dropped if not.
fn read_scalar(reader: TextReader, message: Message, spec: Spec?, name: str) -> void {
    let place = reader.place()
    if reader.peek() == "\"" or reader.peek() == "'" {
        let text = reader.read_string()
        if spec is null { return }
        let field: Spec = spec
        if field.kind != "str" {
            throw ValueError(f"{place}: {name} is not a string field")
        }
        message.add_str(field.name, text)
        return
    }
    let token = reader.read_token()
    if spec is null { return }
    let field: Spec = spec
    if field.kind == "str" {
        throw ValueError(f"{place}: {name} is a string field and {token} is not quoted")
    }
    if field.kind == "float" or field.kind == "double" {
        message.add_floats(field.name, [text_float(token, place, name)])
    } elif field.kind == "bool" {
        message.add_ints(field.name, [text_bool(token, place, name)])
    } elif field.kind == "enum" {
        message.add_ints(field.name, [text_enum(token, field, place)])
    } else {
        message.add_ints(field.name, [text_int(token, place, name)])
    }
}

#: The fields of one message, up to `closer` -- or to the end of the file when
#: `closer` is empty, which is the outermost `NetParameter`.
fn parse_body(reader: TextReader, kind: str, closer: str, depth: int) -> Message {
    if depth > MAX_DEPTH {
        throw ValueError(f"{reader.place()}: this prototxt nests more than {MAX_DEPTH} deep")
    }
    let message = Message(kind)
    let table = layout_of(kind)
    loop {
        reader.skip_blank()
        if reader.at >= reader.size {
            if closer.len() == 0 { return message }
            throw ValueError(f"this prototxt ends inside a {kind}, with no closing {closer}")
        }
        let character = reader.peek()
        if closer.len() > 0 and character == closer {
            reader.at += 1
            return message
        }
        if character == "," or character == ";" {
            reader.at += 1
            continue
        }
        if character == "}" or character == ">" {
            throw ValueError(
                f"{reader.place()}: found {character} where " +
                    f"{if closer.len() > 0 { closer } else { 'a field name' }} was expected"
            )
        }
        if character == "[" {
            throw ValueError(f"{reader.place()}: extension fields are not implemented")
        }
        let name = reader.read_name()
        reader.skip_blank()
        var colon = false
        if reader.peek() == ":" {
            reader.at += 1
            colon = true
            reader.skip_blank()
        }
        let opener = reader.peek()
        if opener == "{" or opener == "<" {
            reader.at += 1
            let shut = if opener == "{" { "}" } else { ">" }
            let spec = table.by_name.get(name, null)
            if spec is null or spec.kind != "msg" {
                skip_body(reader, shut, depth + 1)
            } else {
                let field: Spec = spec
                message.add_child(field.name, parse_body(reader, field.of, shut, depth + 1))
            }
            continue
        }
        if not colon {
            throw ValueError(f"{reader.place()}: the field {name} needs a ':' before its value")
        }
        let spec = table.by_name.get(name, null)
        if reader.peek() != "[" {
            read_scalar(reader, message, spec, name)
            continue
        }
        #: The bracketed list form, `dim: [1, 2, 3]`.
        reader.at += 1
        loop {
            reader.skip_blank()
            if reader.peek() == "]" {
                reader.at += 1
                break
            }
            read_scalar(reader, message, spec, name)
            reader.skip_blank()
            if reader.peek() == "," {
                reader.at += 1
                continue
            }
            if reader.peek() == "]" {
                reader.at += 1
                break
            }
            throw ValueError(f"{reader.place()}: expected ',' or ']' in {name}'s list")
        }
    }
}

#: Read a `NetParameter` written in protobuf's text format.
#:
#: This is the prototxt parser on its own, for a caller that wants the file's
#: fields rather than a graph. Unknown fields are skipped; a syntax error names
#: the line.
pub fn parse_net_text(text: str) -> Message {
    return parse_body(TextReader(text), "NetParameter", "", 0)
}
