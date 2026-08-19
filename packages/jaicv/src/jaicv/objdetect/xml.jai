#: Just enough XML to read what OpenCV writes.
#:
#: OpenCV stores a trained cascade as XML through its own `FileStorage`, which
#: emits a small, regular subset: elements, attributes, text, and nothing else
#: — no namespaces, no entities beyond the five named ones, no processing
#: instructions past the declaration. This reads that subset and refuses
#: anything outside it rather than guessing.
#:
#: ```
#: let root = parse_xml(Path("cascade.xml").read_text())
#: let stages = root.find("cascade").find("stages")
#: for stage in stages.children { ... }
#: ```
from std.core import Printable

#: One element: its name, its attributes, the elements inside it, and the text
#: between them.
pub class Node: Printable {
    #: The tag name.
    pub let name: str
    #: The attributes, in the order they were written.
    pub var attributes: dict[str, str]
    #: The elements directly inside this one.
    pub var children: list[Node]
    #: The text directly inside this one, with the surrounding space trimmed.
    pub var text: str

    pub fn init(self, name: str) {
        self.name = name
        self.attributes = {}
        self.children = []
        self.text = ""
    }

    #: The first child with this name, or `null`.
    pub fn find(self, name: str) -> Node? {
        for child in self.children {
            if child.name == name { return child }
        }
        return null
    }

    #: Every child with this name.
    pub fn find_all(self, name: str) -> list[Node] {
        var out: list[Node] = []
        for child in self.children {
            if child.name == name { out.push(child) }
        }
        return out
    }

    #: The text of the first child with this name, or `fallback`.
    pub fn text_of(self, name: str, fallback: str = "") -> str {
        let child = self.find(name)
        if child is null { return fallback }
        return child.text
    }

    #: The text of the first child with this name, read as a number.
    pub fn number_of(self, name: str, fallback: float = 0.0) -> float {
        let child = self.find(name)
        if child is null { return fallback }
        if child.text.len() == 0 { return fallback }
        return float(child.text)
    }

    #: This element's text split into numbers, which is how OpenCV writes a
    #: matrix or a list of coefficients.
    pub fn numbers(self) -> list[float] {
        var out: list[float] = []
        for piece in self.text.split(" ") {
            let trimmed = piece.strip()
            if trimmed.len() == 0 { continue }
            out.push(float(trimmed))
        }
        return out
    }

    pub fn to_str(self) -> str {
        return f"Node({self.name}, {self.children.len()} children)"
    }

    fn __str__(self) -> str { return self.to_str() }
}

fn unescape(text: str) -> str {
    if not text.contains("&") { return text }
    var out = text
    out = out.replace("&lt;", "<")
    out = out.replace("&gt;", ">")
    out = out.replace("&quot;", "\"")
    out = out.replace("&apos;", "'")
    out = out.replace("&amp;", "&")
    return out
}

fn is_space(character: str) -> bool {
    return character == " " or character == "\t" or character == "\n" or character == "\r"
}

#: Read an XML document and return its root element.
pub fn parse_xml(source: str) -> Node {
    var at = 0
    let length = source.len()
    var stack: list[Node] = []
    var root: Node? = null
    var text = ""

    while at < length {
        if source[at] != "<" {
            #: Text between tags belongs to whichever element is open.
            let start = at
            while at < length and source[at] != "<" { at += 1 }
            text += source[start:at]
            continue
        }

        if at + 3 < length and source[at:at + 4] == "<!--" {
            let close = source.find("-->", at)
            if close < 0 { throw ValueError("this XML has a comment that never closes") }
            at = close + 3
            continue
        }
        if at + 1 < length and source[at + 1] == "?" {
            let close = source.find("?>", at)
            if close < 0 { throw ValueError("this XML has a declaration that never closes") }
            at = close + 2
            continue
        }

        let close = source.find(">", at)
        if close < 0 { throw ValueError("this XML has a tag that never closes") }
        let inside = source[at + 1:close]
        at = close + 1

        if inside.starts_with("/") {
            if stack.len() == 0 {
                throw ValueError(f"this XML closes <{inside[1:]}> with nothing open")
            }
            let finished = stack.pop()
            let wanted = inside[1:].strip()
            if finished.name != wanted {
                throw ValueError(f"this XML closes <{wanted}> where <{finished.name}> was open")
            }
            finished.text = unescape(text).strip()
            text = ""
            if stack.len() == 0 { root = finished }
            continue
        }

        let empty = inside.ends_with("/")
        let body = if empty { inside[0:inside.len() - 1] } else { inside }
        let (name, attributes) = parse_tag(body)
        let node = Node(name)
        node.attributes = attributes
        if stack.len() > 0 {
            #: Text seen before a child belongs to the parent, and a parent
            #: with children never has meaningful text in these files, so the
            #: pending run is simply discarded here.
            stack[stack.len() - 1].children.push(node)
        }
        text = ""
        if empty {
            if stack.len() == 0 { root = node }
        } else {
            stack.push(node)
        }
    }

    if stack.len() > 0 { throw ValueError(f"this XML leaves <{stack[stack.len() - 1].name}> open") }
    let result = root
    if result is null { throw ValueError("this XML has no elements in it") }
    return result
}

fn parse_tag(body: str) -> tuple[str, dict[str, str]] {
    var at = 0
    let length = body.len()
    while at < length and not is_space(body[at]) { at += 1 }
    let name = body[0:at]
    if name.len() == 0 { throw ValueError("this XML has a tag with no name") }

    var attributes: dict[str, str] = {}
    while at < length {
        while at < length and is_space(body[at]) { at += 1 }
        if at >= length { break }
        let key_start = at
        while at < length and body[at] != "=" and not is_space(body[at]) { at += 1 }
        let key = body[key_start:at]
        while at < length and is_space(body[at]) { at += 1 }
        if at >= length or body[at] != "=" {
            attributes[key] = ""
            continue
        }
        at += 1
        while at < length and is_space(body[at]) { at += 1 }
        if at >= length { break }
        let quote = body[at]
        if quote != "\"" and quote != "'" {
            let value_start = at
            while at < length and not is_space(body[at]) { at += 1 }
            attributes[key] = unescape(body[value_start:at])
            continue
        }
        at += 1
        let value_start = at
        while at < length and body[at] != quote { at += 1 }
        attributes[key] = unescape(body[value_start:at])
        at += 1
    }
    return (name, attributes)
}
