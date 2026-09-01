#: The layout decisions the printer defers to: the column limit, what a construct
#: renders as, and where a line may be broken.
from jaithon.ast import Node, NodeKind, TypeExpr, TypeKind, Visibility
from jaithon.ast import op_text
from .text import _INDENT, _skip_string

#: The column the formatter tries to keep every line inside.
#:
#: A soft limit — a long string literal or operator chain may exceed it. Only
#: argument and element lists are broken to obey it.
pub const LINE_WIDTH: int = 100

pub class _Member {
    pub let role: str
    pub let node: any
    pub let start: int

    pub fn init(self, role: str, node: any, start: int) {
        self.role = role
        self.node = node
        self.start = start
    }
}

pub fn _class_members(node: Node) -> list[_Member] {
    let entries: list[_Member] = []
    for field in node.records("fields") {
        entries.push(_Member("field", field, field["span"].start))
    }
    for method in node.child_list("methods") {
        entries.push(_Member("method", method, method.span.start))
    }
    for getter in node.child_list("getters") {
        entries.push(_Member("getter", getter, getter.span.start))
    }
    for setter in node.child_list("setters") {
        entries.push(_Member("setter", setter, setter.span.start))
    }
    return entries
}

pub fn _method_members(node: Node) -> list[_Member] {
    return [_Member("method", method, method.span.start) for method in node.child_list("methods")]
}

pub fn _by_position(entries: list[_Member]) -> list[_Member] {
    let ordered = [entry for entry in entries]
    var index = 1
    while index < ordered.len() {
        let current = ordered[index]
        var position = index - 1
        while position >= 0 and ordered[position].start > current.start {
            ordered[position + 1] = ordered[position]
            position -= 1
        }
        ordered[position + 1] = current
        index += 1
    }
    return ordered
}

pub fn _broken_chain(node: Node, flat: str) -> str? {
    let pieces = _split_operator(flat, op_text(node.op()))
    if pieces.len() < 2 { return null }
    var out = pieces[0]
    for index in 1..pieces.len() {
        out = out + " " + op_text(node.op()) + "\n" + _INDENT + pieces[index]
    }
    return out
}

pub fn _split_operator(text: str, operator: str) -> list[str] {
    let needle = " " + operator + " "
    let chars = [character for character in text]
    let pieces: list[str] = []
    var depth = 0
    var start = 0
    var index = 0
    while index < chars.len() {
        let character = chars[index]
        if character == "\"" or character == "'" {
            let raw = index > 0 and chars[index - 1] == "r"
            let (next, _line, _column) = _skip_string(chars, index, 1, 1, raw)
            index = max(next, index + 1)
            continue
        }
        if character == "(" or character == "[" or character == "{" { depth += 1 }
        if character == ")" or character == "]" or character == "}" { depth -= 1 }
        if depth == 0 and character == " " and _matches_at(chars, index, needle) {
            pieces.push("".join(chars[start:index]))
            index += needle.len()
            start = index
            continue
        }
        index += 1
    }
    pieces.push("".join(chars[start:chars.len()]))
    return pieces
}

fn _matches_at(chars: list[str], index: int, needle: str) -> bool {
    if index + needle.len() > chars.len() { return false }
    for offset in 0..needle.len() {
        if chars[index + offset] != needle[offset] { return false }
    }
    return true
}

pub fn _collapse(lines: list[str]) -> str? {
    if lines.len() < 2 { return null }
    var out = ""
    for raw in lines {
        let part = raw.strip()
        if part.len() == 0 { return null }
        if out.len() == 0 {
            out = part
        } elif part.starts_with("}") or out.ends_with("{") {
            out = out + " " + part
        } else {
            return null
        }
    }
    return out
}

pub fn _is_literal(kind: NodeKind) -> bool {
    return kind == NodeKind.IntLit or
        kind == NodeKind.FloatLit or
        kind == NodeKind.StrLit or
        kind == NodeKind.FString
}

pub fn _is_declaration(node: Node) -> bool {
    let kind = node.kind
    return kind == NodeKind.FnDecl or
        kind == NodeKind.ClassDecl or
        kind == NodeKind.TraitDecl or
        kind == NodeKind.EnumDecl or
        kind == NodeKind.TypeDecl
}

# Statements whose interior comments the printer places by walking a block.
pub fn _has_block(node: Node) -> bool {
    let kind = node.kind
    return kind == NodeKind.If or
        kind == NodeKind.While or
        kind == NodeKind.Loop or
        kind == NodeKind.For or
        kind == NodeKind.Match or
        kind == NodeKind.Try or
        kind == NodeKind.Defer or
        kind == NodeKind.Block or
        kind == NodeKind.FnDecl or
        kind == NodeKind.ClassDecl or
        kind == NodeKind.TraitDecl or
        kind == NodeKind.EnumDecl
}

# An absent annotation and an `Infer` annotation are the same thing written two
# ways, and both print as nothing.
pub fn _type_source(type_expr: TypeExpr?) -> str {
    if type_expr is null { return "" }
    if type_expr.kind == TypeKind.Infer { return "" }
    return type_expr.to_source()
}

pub fn _visibility(value: Visibility) -> str {
    if value == Visibility.Public { return "pub " }
    if value == Visibility.Protected { return "prot " }
    return ""
}

pub fn _label_prefix(label: str?) -> str {
    if label is null { return "" }
    return f"'{label}: "
}

pub fn _label_suffix(label: str?) -> str {
    if label is null { return "" }
    return f" '{label}"
}
