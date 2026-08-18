#: Non-source encodings of a `Node`: the debug s-expression printer and the
#: JSON wire format, split out of ast.jai.
#:
#: `to_sexp` is what `jaithon ast` and the REPL's `:ast` print; `to_json` and
#: `from_json` are the C's `jaiAstToJson` format, the one `jaithon --emit=ast`
#: writes and the self-hosted front end reads back (`from_json(to_json(node))
#: == node`, ast.jai's second property). Both walk a node by its field tags
#: rather than by C-like switch-per-kind (`to_sexp` matches on `NodeKind` for
#: its per-kind attributes, `to_json` is fully tag-driven), and both need the
#: C spelling of `Visibility`, `DeclKind`, `CompKind` and `TypeKind` values,
#: so the four name tables for those live here rather than in ast.jai: nothing
#: there reads them.
#:
#: ```
#: to_sexp(Node.int_lit(1))                              # "(int_lit 1)"
#: from_json(to_json(Node.ident("x"))).text("name")      # "x"
#: ```
import std.json
from jaithon.ast_unparse import at_or_null
from std.math import INF, NAN, is_inf, is_nan
from std.str import StringBuilder, quote
from jaithon.ast import CompKind, DeclKind, Node, NodeKind, OpKind, Span, TypeExpr, TypeKind, Visibility
from jaithon.ast import _ELEMENT_TAG, _RECORDS, _default_field, fields_of, kind_from_name, kind_name
from jaithon.ast import op_from_name, op_name, op_text

let _VISIBILITY_NAMES: dict[Visibility, str] = {
    Visibility.Private: "AST_VIS_PRIVATE",
    Visibility.Protected: "AST_VIS_PROTECTED",
    Visibility.Public: "AST_VIS_PUBLIC",
}

let _DECL_KIND_NAMES: dict[DeclKind, str] = {
    DeclKind.Let: "VD_LET",
    DeclKind.Var: "VD_VAR",
    DeclKind.Const: "VD_CONST",
}

let _COMP_KIND_NAMES: dict[CompKind, str] = {
    CompKind.List: "COMP_LIST",
    CompKind.Dict: "COMP_DICT",
    CompKind.Set: "COMP_SET",
    CompKind.Generator: "COMP_GENERATOR",
}

let _TYPE_KIND_NAMES: dict[TypeKind, str] = {
    TypeKind.Name: "TYPE_NAME",
    TypeKind.Generic: "TYPE_GENERIC",
    TypeKind.Optional: "TYPE_OPTIONAL",
    TypeKind.Union: "TYPE_UNION",
    TypeKind.Fn: "TYPE_FN",
    TypeKind.Tuple: "TYPE_TUPLE",
    TypeKind.Infer: "TYPE_INFER",
    TypeKind.Const: "TYPE_CONST",
}

let _VISIBILITY_BY_NAME: dict[str, Visibility] = {name: value for (value, name) in _VISIBILITY_NAMES.items()}
let _DECL_KIND_BY_NAME: dict[str, DeclKind] = {name: value for (value, name) in _DECL_KIND_NAMES.items()}
let _COMP_KIND_BY_NAME: dict[str, CompKind] = {name: value for (value, name) in _COMP_KIND_NAMES.items()}
let _TYPE_KIND_BY_NAME: dict[str, TypeKind] = {name: value for (value, name) in _TYPE_KIND_NAMES.items()}

# ------------------------------------------------------------------
# S-expression printing
# ------------------------------------------------------------------
fn _emit(out: StringBuilder, text: str) -> void {
    out.push(text)
}

fn _indent(out: StringBuilder, depth: int) -> void {
    var at = 0
    while at < depth {
        out.push("  ")
        at += 1
    }
}

#: `NodeKind.IntLit` -> `"int_lit"`: the C tag without its `AST_` prefix.
fn _kind_tag(kind: NodeKind) -> str {
    var name = kind_name(kind)
    if name.starts_with("AST_") { name = name[4:] }
    return name.lower()
}

fn _quoted(text: str) -> str {
    let out = StringBuilder()
    out.push("\"")
    for character in text {
        let code = ord(character)
        if character == "\"" {
            out.push("\\\"")
        } elif character == "\\" {
            out.push("\\\\")
        } elif character == "\n" {
            out.push("\\n")
        } elif character == "\r" {
            out.push("\\r")
        } elif character == "\t" {
            out.push("\\t")
        } elif code < 0x20 {
            out.push(f"\\x{_hex_byte(code)}")
        } else {
            out.push(character)
        }
    }
    out.push("\"")
    return out.build()
}

fn _hex_byte(value: int) -> str {
    let digits = "0123456789abcdef"
    return digits[value // 16] + digits[value % 16]
}

fn _float_text(value: float) -> str {
    if is_nan(value) { return "nan" }
    if is_inf(value) {
        if value > 0.0 { return "inf" }
        return "-inf"
    }
    return f"{value:.17g}"
}

fn _name_attr(out: StringBuilder, key: str, name: str?) -> void {
    if name is null { return }
    out.push(f" {key}={name}")
}

#: The written form of a type annotation, or `_` when there is none.
fn _type_text(written: TypeExpr?) -> str {
    if written is null { return "_" }
    match written.kind {
        TypeKind.Name => { return written.name ?? "_" }
        TypeKind.Generic => {
            let head = written.name ?? "_"
            return f"{head}[{_type_list(written.args)}]"
        }
        TypeKind.Optional => { return _type_text(written.inner) + "?" }
        TypeKind.Union => {
            var parts: list[str] = []
            for argument in written.args { parts.push(_type_text(argument)) }
            return " | ".join(parts)
        }
        TypeKind.Fn => { return f"fn({_type_list(written.args)}) -> {_type_text(written.inner)}" }
        TypeKind.Tuple => { return f"({_type_list(written.args)})" }
        TypeKind.Infer => { return "_" }
        TypeKind.Const => { return written.name ?? "0" }
    }
}

fn _type_list(types: list[TypeExpr]) -> str {
    var parts: list[str] = []
    for argument in types { parts.push(_type_text(argument)) }
    return ", ".join(parts)
}

fn _type_attr(out: StringBuilder, key: str, written: TypeExpr?) -> void {
    if written is null { return }
    out.push(f" {key}={_type_text(written)}")
}

fn _child(out: StringBuilder, child: Node?, depth: int) -> void {
    out.push("\n")
    if child is null {
        _indent(out, depth)
        out.push("nil")
    } else {
        _print_node(out, child, depth)
    }
}

fn _children(out: StringBuilder, nodes: list[Node], depth: int) -> void {
    for child in nodes { _child(out, child, depth) }
}

fn _params(out: StringBuilder, params: list[dict[str, any]], depth: int) -> void {
    for param in params {
        out.push("\n")
        _indent(out, depth)
        out.push("(param")
        _name_attr(out, "name", _record_str(param, "name"))
        _type_attr(out, "type", _record_type(param, "type"))
        if _record_flag(param, "isVariadic") { out.push(" variadic") }
        if _record_flag(param, "isKwRest") { out.push(" kwrest") }
        let default_value = _record_node(param, "defaultValue")
        if default_value is not null { _child(out, default_value, depth + 1) }
        out.push(")")
    }
}

fn _generics(out: StringBuilder, generics: list[dict[str, any]], depth: int) -> void {
    for generic in generics {
        out.push("\n")
        _indent(out, depth)
        out.push("(generic")
        _name_attr(out, "name", _record_str(generic, "name"))
        _type_attr(out, "bound", _record_type(generic, "bound"))
        out.push(")")
    }
}

fn _record_str(record: dict[str, any], key: str) -> str? {
    let value = record.get(key, null)
    if value is str { return value }
    return null
}

fn _record_node(record: dict[str, any], key: str) -> Node? {
    let value = record.get(key, null)
    if value is Node { return value }
    return null
}

fn _record_nodes(record: dict[str, any], key: str) -> list[Node] {
    let value = record.get(key, null)
    if value is list { return value }
    return []
}

fn _record_type(record: dict[str, any], key: str) -> TypeExpr? {
    let value = record.get(key, null)
    if value is TypeExpr { return value }
    return null
}

fn _record_flag(record: dict[str, any], key: str) -> bool {
    let value = record.get(key, null)
    if value is bool { return value }
    return false
}

fn _record_int(record: dict[str, any], key: str) -> int {
    let value = record.get(key, null)
    if value is int { return value }
    return 0
}

fn _print_body(out: StringBuilder, node: Node, child: int) -> void {
    match node.kind {
        NodeKind.IntLit => { _emit(out, " " + str(node.number("intLit"))) }

        NodeKind.FloatLit => { _emit(out, " " + _float_text(node.get("floatLit"))) }

        NodeKind.StrLit => { _emit(out, " " + _quoted(node.opt_text("chars") ?? "")) }

        NodeKind.BoolLit => { _emit(out, if node.flag("boolLit") { " true" } else { " false" }) }

        NodeKind.NullLit | NodeKind.Self | NodeKind.Super => { }

        NodeKind.FString => { _children(out, node.child_list("parts"), child) }

        NodeKind.Ident => { _name_attr(out, "name", node.opt_text("name")) }

        NodeKind.ListLit | NodeKind.SetLit | NodeKind.TupleLit => {
            _children(out, node.child_list("items"), child)
        }

        NodeKind.DictLit => {
            for (key, value) in zip(node.child_list("keys"), node.child_list("values")) {
                _child(out, key, child)
                _child(out, value, child)
            }
        }

        NodeKind.Unary => {
            _emit(out, f" op={op_text(node.op())}")
            _child(out, node.child("operand"), child)
        }

        NodeKind.Binary | NodeKind.Logical => {
            _emit(out, f" op={op_text(node.op())}")
            _child(out, node.child("left"), child)
            _child(out, node.child("right"), child)
        }

        NodeKind.CompareChain => {
            _emit(out, " ops=")
            for (index, op) in node.get("ops").enumerate() {
                if index > 0 { _emit(out, ",") }
                _emit(out, op_text(op))
            }
            _children(out, node.child_list("operands"), child)
        }

        NodeKind.Ternary => {
            _child(out, node.child("cond"), child)
            _child(out, node.child("thenExpr"), child)
            _child(out, node.child("elseExpr"), child)
        }

        NodeKind.Coalesce => {
            _child(out, node.child("left"), child)
            _child(out, node.child("right"), child)
        }

        NodeKind.Call => {
            _child(out, node.child("callee"), child)
            for argument in node.records("args") {
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(arg")
                _name_attr(out, "name", _record_str(argument, "name"))
                if _record_flag(argument, "isSpread") { _emit(out, " spread") }
                _child(out, _record_node(argument, "value"), child + 1)
                _emit(out, ")")
            }
        }

        NodeKind.Index => {
            _child(out, node.child("object"), child)
            _child(out, node.child("index"), child)
        }

        NodeKind.Slice => {
            _child(out, node.child("object"), child)
            _child(out, node.child("start"), child)
            _child(out, node.child("stop"), child)
            _child(out, node.child("step"), child)
        }

        NodeKind.Member | NodeKind.OptMember => {
            _name_attr(out, "name", node.opt_text("name"))
            _child(out, node.child("object"), child)
        }

        NodeKind.Lambda | NodeKind.AnonFn => {
            _name_attr(out, "name", node.opt_text("name"))
            _type_attr(out, "returns", node.type_field("returnType"))
            let visibility = node.visibility()
            if visibility != Visibility.Private {
                _emit(out, f" vis={_VISIBILITY_NAMES[visibility]}")
            }
            if node.flag("isStatic") { _emit(out, " static") }
            if node.flag("isGenerator") { _emit(out, " generator") }
            if node.flag("isAsync") { _emit(out, " async") }
            _generics(out, node.records("generics"), child)
            _params(out, node.records("params"), child)
            _child(out, node.child("body"), child)
        }

        NodeKind.Comprehension => {
            let comp_kind = node.get("compKind")
            _emit(out, f" kind={_COMP_KIND_NAMES[comp_kind]}")
            _child(out, node.child("element"), child)
            _child(out, node.child("keyExpr"), child)  # dict comprehensions only
            for clause in node.records("clauses") {
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(clause")
                _child(out, _record_node(clause, "pattern"), child + 1)
                _child(out, _record_node(clause, "iterable"), child + 1)
                _children(out, _record_nodes(clause, "conditions"), child + 1)
                _emit(out, ")")
            }
        }

        NodeKind.Range => {
            if node.flag("inclusive") { _emit(out, " inclusive") }
            _child(out, node.child("start"), child)
            _child(out, node.child("stop"), child)
        }

        NodeKind.IfExpr => {
            _child(out, node.child("cond"), child)
            _child(out, node.child("thenBranch"), child)
            _child(out, node.child("elseBranch"), child)
        }

        NodeKind.Match | NodeKind.MatchExpr => {
            _child(out, node.child("subject"), child)
            for arm in node.records("arms") {
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(arm")
                _child(out, _record_node(arm, "pattern"), child + 1)
                _child(out, _record_node(arm, "guard"), child + 1)
                _child(out, _record_node(arm, "body"), child + 1)
                _emit(out, ")")
            }
        }

        NodeKind.Cast => {
            _type_attr(out, "target", node.type_field("target"))
            _child(out, node.child("operand"), child)
        }

        NodeKind.Yield | NodeKind.Await | NodeKind.ThrowExpr => {
            _child(out, node.child("operand"), child)
        }

        NodeKind.If => {
            _child(out, node.child("cond"), child)
            _child(out, node.child("thenBranch"), child)
            _child(out, node.child("elseBranch"), child)
        }

        NodeKind.Block | NodeKind.Program => { _children(out, node.child_list("stmts"), child) }

        NodeKind.ExprStmt => { _child(out, node.child("expr"), child) }

        NodeKind.VarDecl => {
            let decl_kind: DeclKind = node.get("varDeclKind")
            _emit(out, " kind=" + _DECL_KIND_NAMES[decl_kind])
            _type_attr(out, "type", node.type_field("declaredType"))
            if node.visibility() != Visibility.Private {
                _emit(out, " vis=" + _VISIBILITY_NAMES[node.visibility()])
            }
            _child(out, node.child("pattern"), child)
            _child(out, node.child("init"), child)
        }

        NodeKind.Assign => {
            if node.flag("isCompound") { _emit(out, " op=" + op_text(node.op()) + "=") }
            _child(out, node.child("target"), child)
            _child(out, node.child("value"), child)
        }

        NodeKind.While | NodeKind.Loop => {
            _name_attr(out, "label", node.opt_text("label"))
            _child(out, node.child("cond"), child)
            _child(out, node.child("body"), child)
        }

        NodeKind.For => {
            _name_attr(out, "label", node.opt_text("label"))
            _child(out, node.child("pattern"), child)
            _child(out, node.child("iterable"), child)
            _child(out, node.child("body"), child)
        }

        NodeKind.Break | NodeKind.Continue => { _name_attr(out, "label", node.opt_text("label")) }

        NodeKind.Return | NodeKind.Throw => { _child(out, node.child("value"), child) }

        NodeKind.Try => {
            _child(out, node.child("body"), child)
            for handler in node.records("catches") {
                let name: str? = handler.get("name", null)
                let types: list[TypeExpr] = handler.get("types", [])
                let body: Node? = handler.get("body", null)
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(catch")
                _name_attr(out, "name", name)
                for (index, caught) in types.enumerate() {
                    _emit(out, index == 0 ? " types=" : " | ")
                    _emit(out, _type_text(caught))
                }
                _child(out, body, child + 1)
                _emit(out, ")")
            }
            _child(out, node.child("finallyBlock"), child)
        }

        NodeKind.Defer => { _child(out, node.child("body"), child) }

        NodeKind.Assert => {
            _child(out, node.child("cond"), child)
            _child(out, node.child("message"), child)
        }

        NodeKind.FnDecl => {
            _name_attr(out, "name", node.opt_text("name"))
            _type_attr(out, "returns", node.type_field("returnType"))
            let vis = node.visibility()
            if vis != Visibility.Private { _emit(out, f" vis={_VISIBILITY_NAMES[vis]}") }
            if node.flag("isStatic") { _emit(out, " static") }
            if node.flag("isGenerator") { _emit(out, " generator") }
            if node.flag("isAsync") { _emit(out, " async") }
            _generics(out, node.records("generics"), child)
            _params(out, node.records("params"), child)
            _child(out, node.child("body"), child)
        }

        NodeKind.ClassDecl => {
            _name_attr(out, "name", node.opt_text("name"))
            _type_attr(out, "extends", node.type_field("superclass"))
            let vis = node.visibility()
            if vis != Visibility.Private { _emit(out, f" vis={_VISIBILITY_NAMES[vis]}") }
            if node.flag("isAbstract") { _emit(out, " abstract") }
            for (index, trait_type) in node.type_list("traits").enumerate() {
                _emit(out, if index == 0 { " traits=" } else { "," })
                _emit(out, _type_text(trait_type))
            }
            _generics(out, node.records("generics"), child)
            for current in node.records("fields") {
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(field")
                _name_attr(out, "name", _record_str(current, "name"))
                _type_attr(out, "type", _record_type(current, "type"))
                let field_vis: Visibility = current.get("visibility", Visibility.Private)
                _emit(out, f" vis={_VISIBILITY_NAMES[field_vis]}")
                if _record_flag(current, "isStatic") { _emit(out, " static") }
                _emit(out, if _record_flag(current, "isLet") { " let" } else { " var" })
                let default_value = _record_node(current, "defaultValue")
                if default_value is not null { _child(out, default_value, child + 1) }
                _emit(out, ")")
            }
            _children(out, node.child_list("methods"), child)
            _children(out, node.child_list("getters"), child)
            _children(out, node.child_list("setters"), child)
        }

        NodeKind.TraitDecl => {
            _name_attr(out, "name", node.opt_text("name"))
            let vis = node.visibility()
            if vis != Visibility.Private { _emit(out, f" vis={_VISIBILITY_NAMES[vis]}") }
            for (index, super_type) in node.type_list("supers").enumerate() {
                _emit(out, if index == 0 { " supers=" } else { "," })
                _emit(out, _type_text(super_type))
            }
            _generics(out, node.records("generics"), child)
            _children(out, node.child_list("methods"), child)
        }

        NodeKind.EnumDecl => {
            _name_attr(out, "name", node.opt_text("name"))
            let vis = node.visibility()
            if vis != Visibility.Private { _emit(out, f" vis={_VISIBILITY_NAMES[vis]}") }
            _generics(out, node.records("generics"), child)
            for current in node.records("variants") {
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(variant")
                _name_attr(out, "name", _record_str(current, "name"))
                _params(out, current.get("params", []), child + 1)
                _emit(out, ")")
            }
            _children(out, node.child_list("methods"), child)
        }

        NodeKind.TypeDecl => {
            _name_attr(out, "name", node.opt_text("name"))
            _type_attr(out, "aliased", node.type_field("aliased"))
            let vis = node.visibility()
            if vis != Visibility.Private { _emit(out, f" vis={_VISIBILITY_NAMES[vis]}") }
            _generics(out, node.records("generics"), child)
        }

        NodeKind.Import => {
            _name_attr(out, "path", node.opt_text("path"))
            _name_attr(out, "alias", node.opt_text("alias"))
        }

        NodeKind.FromImport => {
            _name_attr(out, "path", node.opt_text("path"))
            if node.flag("isWildcard") { _emit(out, " wildcard") }
            for current in node.records("items") {
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(item")
                _name_attr(out, "name", _record_str(current, "name"))
                _name_attr(out, "alias", _record_str(current, "alias"))
                _emit(out, ")")
            }
        }

        NodeKind.Export => {
            let names: list[str?] = node.get("names")
            for (index, name) in names.enumerate() {
                _emit(out, if index == 0 { " names=" } else { "," })
                _emit(out, name ?? "?")
            }
        }

        NodeKind.ModuleDecl => { _name_attr(out, "name", node.opt_text("name")) }

        NodeKind.PatWildcard => { }

        NodeKind.PatBind => {
            _name_attr(out, "name", node.opt_text("name"))
            _type_attr(out, "type", node.type_field("type"))
        }

        NodeKind.PatLiteral => { _child(out, node.child("value"), child) }

        NodeKind.PatRange => {
            if node.flag("inclusive") { _emit(out, " inclusive") }
            _child(out, node.child("lo"), child)
            _child(out, node.child("hi"), child)
        }

        NodeKind.PatTuple | NodeKind.PatList | NodeKind.PatOr => {
            let rest = node.number("restIndex")
            if rest >= 0 { _emit(out, f" rest={rest}") }
            _children(out, node.child_list("elems"), child)
        }

        NodeKind.PatClass | NodeKind.PatEnum => {
            _name_attr(out, "typeName", node.opt_text("typeName"))
            _name_attr(out, "variantName", node.opt_text("variantName"))
            let names = node.get("fieldNames")
            for (index, sub) in node.child_list("subPatterns").enumerate() {
                let name = at_or_null(names, index)
                if name is null {
                    _child(out, sub, child)
                    continue
                }
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, f"(field name={name}")
                _child(out, sub, child + 1)
                _emit(out, ")")
            }
        }
    }
}

fn _print_node(out: StringBuilder, node: Node, depth: int) -> void {
    _indent(out, depth)
    out.push("(")
    out.push(_kind_tag(node.kind))

    let child = depth + 1
    _print_body(out, node, child)
    out.push(")")
}

#: Print `node` the way `jaithon ast` prints it, with no trailing newline.
#:
#: ```
#: to_sexp(Node.int_lit(1))       # "(int_lit 1)"
#: ```
pub fn to_sexp(node: Node) -> str {
    let out = StringBuilder()
    _print_node(out, node, 0)
    return out.build()
}

# ------------------------------------------------------------------
# JSON
# ------------------------------------------------------------------

#: Encode `node` as the JSON `jaiAstToJson` writes.
#:
#: Keys appear in the C's order, so the two front ends' output can be diffed
#: directly. Two differences are deliberate: a float may be spelled differently
#: though never valued differently, and the trailing newline the C writes after
#: the document is the caller's to add.
#:
#: ```
#: let encoded = to_json(Node.ident("x"))
#: from_json(encoded).text("name")   # "x"
#: ```
pub fn to_json(node: Node) -> str {
    return json.stringify(to_value(node))
}

#: Decode a node from the JSON `jaiAstToJson` wrote.
#:
#: Raises `ParseError` for malformed JSON and `ValueError` for JSON that is
#: well formed but is not an AST.
pub fn from_json(text: str) -> Node {
    return from_value(json.parse(text))
}

#: Convert `node` to plain dicts, lists and scalars.
#:
#: This is the same shape `to_json` serialises, for tools that would rather walk
#: the data than re-parse it. A null node converts to null.
pub fn to_value(node: Node?) -> any {
    if node is null { return null }
    let out: dict[str, any] = {"kind": kind_name(node.kind), "span": _span_value(node.span)}
    for (name, tag) in fields_of(node.kind) {
        out[name] = _value_of(node.fields.get(name, _default_field(name, tag)), tag)
    }
    return out
}

#: Rebuild a node from the plain data `to_value` produces.
pub fn from_value(value: any) -> Node {
    let node = _node_from(value)
    if node is null { throw ValueError("expected an AST node object, got null") }
    return node
}

fn _value_of(value: any, tag: str) -> any {
    if tag in _ELEMENT_TAG {
        let element = _ELEMENT_TAG[tag]
        return [_value_of(item, element) for item in value]
    }
    if tag == "node" { return to_value(value) }
    if tag == "type" { return _type_value(value) }
    if tag == "span" { return _span_value(value) }
    if tag == "op" { return op_name(value) }
    if tag == "vis" { return _VISIBILITY_NAMES[value] }
    if tag == "vdk" { return _DECL_KIND_NAMES[value] }
    if tag == "compk" { return _COMP_KIND_NAMES[value] }
    if tag == "float" { return _float_value(value) }
    if tag in _RECORDS { return _record_value(value, _RECORDS[tag]) }
    return value
}

fn _record_value(record: dict[str, any], schema: list[tuple[str, str]]) -> dict[str, any] {
    let out: dict[str, any] = {}
    for (name, tag) in schema {
        out[name] = _value_of(record.get(name, _default_field(name, tag)), tag)
    }
    return out
}

fn _span_value(span: Span) -> dict[str, any] {
    return {"start": span.start, "end": span.end, "file": span.file}
}

fn _type_value(type_expr: TypeExpr?) -> any {
    if type_expr is null { return null }
    return {
        "kind": _TYPE_KIND_NAMES[type_expr.kind],
        "span": _span_value(type_expr.span),
        "name": type_expr.name,
        "args": [_type_value(argument) for argument in type_expr.args],
        "inner": _type_value(type_expr.inner),
    }
}

fn _float_value(value: float) -> any {
    if is_nan(value) { return "NaN" }
    if is_inf(value) {
        if value > 0.0 { return "Infinity" }
        return "-Infinity"
    }
    return value
}

fn _node_from(value: any) -> Node? {
    if value is null { return null }
    if not isinstance(value, dict) {
        throw ValueError(f"expected an AST node object, got {repr(value)}")
    }
    let name = value.get("kind", null)
    if name is null { throw ValueError("an AST node object needs a `kind`") }
    let kind = kind_from_name(name)
    let node = Node(kind, span: _span_from(value.get("span", null)))
    for (field, tag) in fields_of(kind) {
        node.fields[field] = _from_value(value.get(field, null), field, tag)
    }
    return node
}

fn _from_value(value: any, name: str, tag: str) -> any {
    if value is null { return _default_field(name, tag) }
    if tag in _ELEMENT_TAG {
        let element = _ELEMENT_TAG[tag]
        return [_from_value(item, name, element) for item in value]
    }
    if tag == "node" { return _node_from(value) }
    if tag == "type" { return _type_from(value) }
    if tag == "span" { return _span_from(value) }
    if tag == "op" { return op_from_name(value) }
    if tag == "vis" { return _enum_from(_VISIBILITY_BY_NAME, value, "visibility") }
    if tag == "vdk" { return _enum_from(_DECL_KIND_BY_NAME, value, "declaration kind") }
    if tag == "compk" { return _enum_from(_COMP_KIND_BY_NAME, value, "comprehension kind") }
    if tag == "float" { return _float_from(value) }
    if tag in _RECORDS { return _record_from(value, _RECORDS[tag]) }
    return value
}

fn _record_from(value: any, schema: list[tuple[str, str]]) -> dict[str, any] {
    if not isinstance(value, dict) {
        throw ValueError(f"expected an AST sub-structure object, got {repr(value)}")
    }
    let record: dict[str, any] = {}
    for (name, tag) in schema {
        record[name] = _from_value(value.get(name, null), name, tag)
    }
    return record
}

fn _span_from(value: any) -> Span {
    if value is null { return Span.none() }
    return Span(value.get("file", -1), value.get("start", 0), value.get("end", 0))
}

fn _type_from(value: any) -> TypeExpr? {
    if value is null { return null }
    let raw = value.get("kind", null)
    let kind = _TYPE_KIND_BY_NAME.get(raw, null)
    if kind is null { throw ValueError(f"unknown AST type kind: {repr(raw)}") }
    let args: list[TypeExpr] = []
    for raw_arg in value.get("args", []) {
        let argument = _type_from(raw_arg)
        if argument is not null { args.push(argument) }
    }
    return TypeExpr(
        kind,
        name: value.get("name", null),
        args: args,
        inner: _type_from(value.get("inner", null)),
        span: _span_from(value.get("span", null))
    )
}

fn _enum_from(table: dict[str, any], name: any, what: str) -> any {
    let value = table.get(name, null)
    if value is null { throw ValueError(f"unknown {what}: {repr(name)}") }
    return value
}

fn _float_from(value: any) -> float {
    if isinstance(value, str) {
        if value == "NaN" { return NAN }
        if value == "Infinity" { return INF }
        if value == "-Infinity" { return -INF }
        throw ValueError(f"expected a float, got the string {quote(value)}")
    }
    return float(value)
}
