#: Rendering a `Node` back to Jaithon source text, split out of ast.jai.
#:
#: `to_source` is the formatter's whole engine: parsing its output yields a
#: tree equal to the original in every field but span (ast.jai's third
#: property). This is by far the largest of the three pieces that moved out
#: of ast.jai -- a precedence-aware printer needs a case for every expression,
#: statement and pattern kind, the same way the parser that reads it back
#: does -- so it gets a file of its own; `tool/fmt.jai` and `tool/test.jai`
#: are its only importers.
#:
#: `_flag`, `_text_at`, `_node_at` and `_type_at` below read a sub-structure
#: record the same way ast.jai's own `_record` helpers do (`always_diverges`
#: uses `_node_at` there); they are re-declared here rather than imported
#: because they are three-line wrappers around `dict.get`, not schema this
#: module and ast.jai both need to agree on.
#:
#: ```
#: let sum = Node.binary(OpKind.Add, Node.int_lit(1), Node.int_lit(2))
#: to_source(Node.binary(OpKind.Mul, sum, Node.int_lit(3)))   # "(1 + 2) * 3"
#: ```
from jaithon.ast import CompKind, DeclKind, Node, NodeKind, OpKind, TypeExpr, TypeKind, Visibility
from jaithon.ast import is_expression, is_pattern, kind_name, op_text
from std.str import StringBuilder, escape, quote, repeat

# Typed reads of a sub-structure's fields. Private copies of ast.jai's own
# `_flag`/`_text_at`/`_node_at`/`_type_at` -- see the file comment above.
fn _flag(record: dict[str, any], name: str) -> bool {
    let value: bool = record.get(name, false)
    return value
}

fn _text_at(record: dict[str, any], name: str) -> str? {
    let value: str? = record.get(name, null)
    return value
}

fn _node_at(record: dict[str, any], name: str) -> Node? {
    let value: Node? = record.get(name, null)
    return value
}

fn _type_at(record: dict[str, any], name: str) -> TypeExpr? {
    let value: TypeExpr? = record.get(name, null)
    return value
}

# Precedence levels, from §4.1 of the language spec. A child is parenthesised
# exactly when its level is below the level its position requires, which is what
# makes `to_source` round-trip without wrapping everything in brackets.
const _P_ATOM = 17
const _P_POSTFIX = 16
const _P_POW = 15
const _P_UNARY = 14
const _P_MUL = 13
const _P_ADD = 12
const _P_SHIFT = 11
const _P_BAND = 10
const _P_BXOR = 9
const _P_BOR = 8
const _P_RANGE = 7
const _P_COMPARE = 6
const _P_NOT = 5
const _P_AND = 4
const _P_OR = 3
const _P_COALESCE = 2
const _P_LOWEST = 1

let _OP_PRECEDENCE: dict[OpKind, int] = {
    OpKind.Pow: _P_POW,
    OpKind.Mul: _P_MUL,
    OpKind.Div: _P_MUL,
    OpKind.FloorDiv: _P_MUL,
    OpKind.Mod: _P_MUL,
    OpKind.MatMul: _P_MUL,
    OpKind.MulWrap: _P_MUL,
    OpKind.Add: _P_ADD,
    OpKind.Sub: _P_ADD,
    OpKind.AddWrap: _P_ADD,
    OpKind.SubWrap: _P_ADD,
    OpKind.Shl: _P_SHIFT,
    OpKind.Shr: _P_SHIFT,
    OpKind.Band: _P_BAND,
    OpKind.Bxor: _P_BXOR,
    OpKind.Bor: _P_BOR,
    OpKind.Eq: _P_COMPARE,
    OpKind.Ne: _P_COMPARE,
    OpKind.Lt: _P_COMPARE,
    OpKind.Le: _P_COMPARE,
    OpKind.Gt: _P_COMPARE,
    OpKind.Ge: _P_COMPARE,
    OpKind.Is: _P_COMPARE,
    OpKind.IsNot: _P_COMPARE,
    OpKind.In: _P_COMPARE,
    OpKind.NotIn: _P_COMPARE,
    OpKind.And: _P_AND,
    OpKind.Or: _P_OR,
    OpKind.Neg: _P_UNARY,
    OpKind.Pos: _P_UNARY,
    OpKind.Bnot: _P_UNARY,
    OpKind.Not: _P_NOT,
}

let _DECL_KEYWORDS: dict[DeclKind, str] = {
    DeclKind.Let: "let",
    DeclKind.Var: "var",
    DeclKind.Const: "const",
}

# Top-level declarations that read better with a blank line around them.
let _BLOCK_DECLARATIONS: set[NodeKind] = {
    NodeKind.FnDecl,
    NodeKind.ClassDecl,
    NodeKind.TraitDecl,
    NodeKind.EnumDecl,
}

#: Render `node` as Jaithon source.
#:
#: Parsing the result yields a tree equal to `node` in every field except its
#: spans, which move because the text moved. That property is what makes the
#: formatter possible: it parses, prints, and trusts that nothing but layout
#: changed. Three details are worth knowing:
#:
#: - `Cast` nodes are inserted by the type checker and have no surface syntax,
#:   so they print as their operand.
#: - `isAbstract`, `isGenerator` and `isAsync` are inferred by the parser rather
#:   than written, so they carry no output of their own.
#: - An empty set literal prints as `set()`, since `{}` is the empty dict.
#:
#: ```
#: let sum = Node.binary(OpKind.Add, Node.int_lit(1), Node.int_lit(2))
#: to_source(Node.binary(OpKind.Mul, sum, Node.int_lit(3)))   # "(1 + 2) * 3"
#: ```
pub fn to_source(node: Node) -> str {
    return _Printer().render(node)
}

fn _precedence(node: Node) -> int {
    return match node.kind {
        NodeKind.Binary | NodeKind.Logical | NodeKind.Unary => _OP_PRECEDENCE[node.op()],
        NodeKind.CompareChain => _P_COMPARE,
        NodeKind.Range => _P_RANGE,
        NodeKind.Coalesce => _P_COALESCE,
        NodeKind.Call | NodeKind.Index | NodeKind.Slice | NodeKind.Member | NodeKind.OptMember => _P_POSTFIX,
        NodeKind.Ternary | NodeKind.Lambda | NodeKind.IfExpr | NodeKind.MatchExpr | NodeKind.ThrowExpr | NodeKind.Yield | NodeKind.Await => _P_LOWEST,
        NodeKind.Cast => _precedence(node.require("operand")),
        _ => _P_ATOM,
    }
}

class _Printer {
    let out: StringBuilder
    var indent: int
    var indent_pending: bool

    fn init(self) {
        self.out = StringBuilder()
        self.indent = 0
        self.indent_pending = false
    }

    #: Render `node` and return the text. The one entry point; everything else
    #: here is internal to the printer.
    pub fn render(self, node: Node) -> str {
        if node.kind == NodeKind.Program {
            self.program(node)
        } elif is_pattern(node.kind) {
            self.pattern(node)
        } elif is_expression(node.kind) {
            self.expr(node, _P_LOWEST)
        } else {
            self.stmt(node)
        }
        return self.out.build()
    }

    fn emit(self, text: str) -> void {
        if self.indent_pending {
            self.out.push(repeat("    ", self.indent))
            self.indent_pending = false
        }
        self.out.push(text)
    }

    # A blank line stays blank: the indent is only written once something else is.
    fn line(self) -> void {
        self.out.push("\n")
        self.indent_pending = true
    }

    fn program(self, node: Node) -> void {
        let stmts = node.child_list("stmts")
        for (index, stmt) in stmts.enumerate() {
            if index > 0 and (_is_block_decl(stmts[index - 1]) or _is_block_decl(stmt)) {
                self.line()
            }
            self.stmt(stmt)
            self.line()
        }
    }

    fn block(self, node: Node) -> void {
        let stmts = node.child_list("stmts")
        if stmts.len() == 0 {
            self.emit("{}")
            return
        }
        self.emit("{")
        self.line()
        self.indent += 1
        for stmt in stmts {
            self.stmt(stmt)
            self.line()
        }
        self.indent -= 1
        self.emit("}")
    }

    # A branch is a block in every tree the parser builds; anything else still
    # has to print inside braces to parse back as one statement.
    fn body(self, node: Node?) -> void {
        if node is null {
            self.emit("{}")
            return
        }
        if node.kind == NodeKind.Block {
            self.block(node)
            return
        }
        self.emit("{ ")
        self.stmt(node)
        self.emit(" }")
    }

    fn stmt(self, node: Node?) -> void {
        if node is null { return }
        match node.kind {
            NodeKind.Block => { self.block(node) }
            NodeKind.ExprStmt => { self.expr(node.child("expr"), _P_LOWEST) }
            NodeKind.VarDecl => { self.var_decl(node) }
            NodeKind.Assign => { self.assign(node) }
            NodeKind.If => { self.conditional(node) }
            NodeKind.While => {
                self.label(node)
                self.emit("while ")
                self.expr(node.child("cond"), _P_LOWEST)
                self.emit(" ")
                self.body(node.child("body"))
            }
            NodeKind.Loop => {
                self.label(node)
                self.emit("loop ")
                self.body(node.child("body"))
            }
            NodeKind.For => {
                self.label(node)
                self.emit("for ")
                self.pattern(node.child("pattern"))
                self.emit(" in ")
                self.expr(node.child("iterable"), _P_LOWEST)
                self.emit(" ")
                self.body(node.child("body"))
            }
            NodeKind.Match => { self.match_node(node) }
            NodeKind.Break => { self.jump(node, "break") }
            NodeKind.Continue => { self.jump(node, "continue") }
            NodeKind.Return => {
                self.emit("return")
                let value = node.child("value")
                if value is not null {
                    self.emit(" ")
                    self.expr(value, _P_LOWEST)
                }
            }
            NodeKind.Throw => {
                self.emit("throw ")
                self.expr(node.child("value"), _P_LOWEST)
            }
            NodeKind.Try => { self.try_stmt(node) }
            NodeKind.Defer => {
                self.emit("defer ")
                self.body(node.child("body"))
            }
            NodeKind.Assert => {
                self.emit("assert ")
                self.expr(node.child("cond"), _P_LOWEST)
                let message = node.child("message")
                if message is not null {
                    self.emit(", ")
                    self.expr(message, _P_LOWEST)
                }
            }
            NodeKind.FnDecl => { self.function(node, "fn") }
            NodeKind.ClassDecl => { self.class_decl(node) }
            NodeKind.TraitDecl => { self.trait_decl(node) }
            NodeKind.EnumDecl => { self.enum_decl(node) }
            NodeKind.TypeDecl => {
                self.visibility(node.visibility())
                self.emit("type " + node.text("name"))
                self.generics(node.records("generics"))
                self.emit(" = " + _type_source(node.type_field("aliased")))
            }
            NodeKind.Import => {
                self.emit("import " + node.text("path"))
                let alias = node.opt_text("alias")
                if alias is not null { self.emit(" as " + alias) }
            }
            NodeKind.FromImport => { self.from_import(node) }
            NodeKind.Export => { self.emit("export { " + ", ".join(node.get("names")) + " }") }
            NodeKind.ModuleDecl => { self.emit("module " + node.text("name")) }
            NodeKind.Program => { self.program(node) }
            _ => { self.expr(node, _P_LOWEST) }
        }
    }

    fn label(self, node: Node) -> void {
        let label = node.opt_text("label")
        if label is not null { self.emit("'" + label + ": ") }
    }

    fn jump(self, node: Node, keyword: str) -> void {
        self.emit(keyword)
        let label = node.opt_text("label")
        if label is not null { self.emit(" '" + label) }
    }

    fn visibility(self, value: Visibility) -> void {
        if value == Visibility.Public {
            self.emit("pub ")
        } elif value == Visibility.Protected {
            self.emit("prot ")
        }
    }

    fn var_decl(self, node: Node) -> void {
        self.visibility(node.visibility())
        self.emit(_DECL_KEYWORDS[node.get("varDeclKind")] + " ")
        self.pattern(node.child("pattern"))
        let declared = node.type_field("declaredType")
        if declared is not null and declared.kind != TypeKind.Infer {
            self.emit(": " + declared.to_source())
        }
        let init = node.child("init")
        if init is not null {
            self.emit(" = ")
            self.expr(init, _P_LOWEST)
        }
    }

    fn assign(self, node: Node) -> void {
        let target = node.child("target")
        if target is not null and is_pattern(target.kind) {
            self.pattern(target)
        } else {
            self.expr(target, _P_LOWEST)
        }
        if node.flag("isCompound") {
            self.emit(" " + op_text(node.op()) + "= ")
        } else {
            self.emit(" = ")
        }
        self.expr(node.child("value"), _P_LOWEST)
    }

    # `elif` is an else branch that is itself an `if`; `else { if ... }` keeps
    # its block and so keeps its braces.
    fn conditional(self, node: Node) -> void {
        self.emit("if ")
        self.expr(node.child("cond"), _P_LOWEST)
        self.emit(" ")
        self.body(node.child("thenBranch"))
        var alternative = node.child("elseBranch")
        while alternative is not null {
            if alternative.kind == NodeKind.If or alternative.kind == NodeKind.IfExpr {
                self.emit(" elif ")
                self.expr(alternative.child("cond"), _P_LOWEST)
                self.emit(" ")
                self.body(alternative.child("thenBranch"))
                alternative = alternative.child("elseBranch")
            } else {
                self.emit(" else ")
                self.body(alternative)
                alternative = null
            }
        }
    }

    fn match_node(self, node: Node) -> void {
        self.emit("match ")
        self.expr(node.child("subject"), _P_LOWEST)
        self.emit(" {")
        self.line()
        self.indent += 1
        for current in node.records("arms") {
            self.pattern(_node_at(current, "pattern"))
            let guard = _node_at(current, "guard")
            if guard is not null {
                self.emit(" if ")
                self.expr(guard, _P_LOWEST)
            }
            self.emit(" => ")
            let body = _node_at(current, "body")
            if body is not null and body.kind == NodeKind.Block {
                self.block(body)
            } else {
                self.expr(body, _P_LOWEST)
                self.emit(",")
            }
            self.line()
        }
        self.indent -= 1
        self.emit("}")
    }

    fn try_stmt(self, node: Node) -> void {
        self.emit("try ")
        self.body(node.child("body"))
        for handler in node.records("catches") {
            self.emit(" catch")
            let name = _text_at(handler, "name")
            if name is not null { self.emit(" " + name) }
            let types = handler.get("types", [])
            if types.len() > 0 {
                self.emit(": " + " | ".join([caught.to_source() for caught in types]))
            }
            self.emit(" ")
            self.body(_node_at(handler, "body"))
        }
        let ensure = node.child("finallyBlock")
        if ensure is not null {
            self.emit(" finally ")
            self.body(ensure)
        }
    }

    fn from_import(self, node: Node) -> void {
        self.emit("from " + node.text("path") + " import ")
        if node.flag("isWildcard") {
            self.emit("*")
            return
        }
        for (index, item) in node.records("items").enumerate() {
            if index > 0 { self.emit(", ") }
            self.emit(_text_at(item, "name") ?? "_")
            let alias = _text_at(item, "alias")
            if alias is not null { self.emit(" as " + alias) }
        }
    }

    fn function(self, node: Node, keyword: str) -> void {
        self.visibility(node.visibility())
        if node.flag("isStatic") { self.emit("static ") }
        self.emit(keyword)
        let name = node.opt_text("name")
        if name is not null { self.emit(" " + name) }
        self.generics(node.records("generics"))
        self.params(node.records("params"))
        let returns = node.type_field("returnType")
        if returns is not null { self.emit(" -> " + returns.to_source()) }
        # A trait method without a default implementation is a signature alone.
        let body = node.child("body")
        if body is null { return }
        self.emit(" ")
        self.body(body)
    }

    fn generics(self, items: list[dict[str, any]]) -> void {
        if items.len() == 0 { return }
        self.emit("[")
        for (index, current) in items.enumerate() {
            if index > 0 { self.emit(", ") }
            self.emit(_text_at(current, "name") ?? "_")
            let bound = _type_at(current, "bound")
            if bound is not null { self.emit(": " + bound.to_source()) }
        }
        self.emit("]")
    }

    fn params(self, items: list[dict[str, any]]) -> void {
        self.emit("(")
        for (index, current) in items.enumerate() {
            if index > 0 { self.emit(", ") }
            if _flag(current, "isVariadic") { self.emit("...") }
            if _flag(current, "isKwRest") { self.emit("**") }
            self.emit(_text_at(current, "name") ?? "_")
            let declared = _type_at(current, "type")
            if declared is not null { self.emit(": " + declared.to_source()) }
            let fallback = _node_at(current, "defaultValue")
            if fallback is not null {
                self.emit(" = ")
                self.expr(fallback, _P_LOWEST)
            }
        }
        self.emit(")")
    }

    fn class_decl(self, node: Node) -> void {
        self.visibility(node.visibility())
        self.emit("class " + node.text("name"))
        self.generics(node.records("generics"))
        let superclass = node.type_field("superclass")
        if superclass is not null { self.emit(" extends " + superclass.to_source()) }
        self.implements_list(node.type_list("traits"))
        let fields = node.records("fields")
        let methods = node.child_list("methods")
        let getters = node.child_list("getters")
        let setters = node.child_list("setters")
        if fields.len() + methods.len() + getters.len() + setters.len() == 0 {
            self.emit(" {}")
            return
        }
        self.emit(" {")
        self.line()
        self.indent += 1
        for current in fields {
            self.class_field(current)
            self.line()
        }
        var written = fields.len() > 0
        written = self.members(methods, "fn", written)
        written = self.members(getters, "get", written)
        written = self.members(setters, "set", written)
        self.indent -= 1
        self.emit("}")
    }

    # Members are separated by a blank line; `written` says whether anything is
    # above them already.
    fn members(self, nodes: list[Node], keyword: str, written: bool) -> bool {
        var any_written = written
        for member in nodes {
            if any_written { self.line() }
            self.function(member, keyword)
            self.line()
            any_written = true
        }
        return any_written
    }

    fn class_field(self, field: dict[str, any]) -> void {
        let value: Visibility = field.get("visibility", Visibility.Private)
        self.visibility(value)
        if _flag(field, "isStatic") { self.emit("static ") }
        if _flag(field, "isLet") {
            self.emit("let ")
        } else {
            self.emit("var ")
        }
        self.emit(_text_at(field, "name") ?? "_")
        let declared = _type_at(field, "type")
        if declared is not null { self.emit(": " + declared.to_source()) }
        let fallback = _node_at(field, "defaultValue")
        if fallback is not null {
            self.emit(" = ")
            self.expr(fallback, _P_LOWEST)
        }
    }

    fn trait_decl(self, node: Node) -> void {
        self.visibility(node.visibility())
        self.emit("trait " + node.text("name"))
        self.generics(node.records("generics"))
        self.implements_list(node.type_list("supers"))
        let methods = node.child_list("methods")
        if methods.len() == 0 {
            self.emit(" {}")
            return
        }
        self.emit(" {")
        self.line()
        self.indent += 1
        let _written = self.members(methods, "fn", false)
        self.indent -= 1
        self.emit("}")
    }

    fn enum_decl(self, node: Node) -> void {
        self.visibility(node.visibility())
        self.emit("enum " + node.text("name"))
        self.generics(node.records("generics"))
        self.emit(" {")
        self.line()
        self.indent += 1
        for current in node.records("variants") {
            self.emit(_text_at(current, "name") ?? "_")
            let params = current.get("params", [])
            if params.len() > 0 { self.params(params) }
            self.emit(",")
            self.line()
        }
        let _written = self.members(node.child_list("methods"), "fn", true)
        self.indent -= 1
        self.emit("}")
    }

    fn implements_list(self, types: list[TypeExpr]) -> void {
        if types.len() == 0 { return }
        self.emit(": " + ", ".join([one.to_source() for one in types]))
    }

    fn expr(self, node: Node?, min_precedence: int) -> void {
        if node is null { return }
        # A sign-leading literal is one token, not an operator and a number
        # (§1.6: `-9223372036854775808` is the only one, since every other
        # negative literal parses as `-` applied to a positive one). Let any
        # operator stand in front of it and that minus binds to the operator
        # instead, leaving digits that overflow `int`: `- -9223372036854775808`
        # and `1 - -9223372036854775808` are both rejected. Brackets are the
        # only spelling that survives in every position, so it always wears
        # them.
        let wrap = _precedence(node) < min_precedence or _is_signed_literal(node)
        if wrap { self.emit("(") }
        self.bare_expr(node)
        if wrap { self.emit(")") }
    }

    # `1.foo` would lex as a float, so an integer receiver keeps its brackets
    # even though its precedence says it needs none.
    fn receiver(self, node: Node?) -> void {
        if node is not null and node.kind == NodeKind.IntLit {
            self.emit("(")
            self.bare_expr(node)
            self.emit(")")
            return
        }
        self.expr(node, _P_POSTFIX)
    }

    fn bare_expr(self, node: Node) -> void {
        match node.kind {
            NodeKind.IntLit => { self.emit(str(node.number("intLit"))) }
            NodeKind.FloatLit => { self.emit(_float_source(node.get("floatLit"))) }
            NodeKind.StrLit => { self.emit(quote(node.opt_text("chars") ?? "")) }
            NodeKind.BoolLit => { self.emit(node.flag("boolLit") ? "true" : "false") }
            NodeKind.NullLit => { self.emit("null") }
            NodeKind.Self => { self.emit("self") }
            NodeKind.Super => { self.emit("super") }
            NodeKind.Ident => { self.emit(node.text("name")) }
            NodeKind.FString => { self.fstring(node) }
            NodeKind.ListLit => { self.sequence(node.child_list("items"), "[", "]", false) }
            NodeKind.SetLit => { self.set_lit(node) }
            NodeKind.TupleLit => { self.sequence(node.child_list("items"), "(", ")", true) }
            NodeKind.DictLit => { self.dict_lit(node) }
            NodeKind.Unary => { self.unary(node) }
            NodeKind.Binary | NodeKind.Logical => { self.binary(node) }
            NodeKind.CompareChain => { self.compare_chain(node) }
            NodeKind.Ternary => {
                self.expr(node.child("cond"), _P_COALESCE)
                self.emit(" ? ")
                self.expr(node.child("thenExpr"), _P_LOWEST)
                self.emit(" : ")
                self.expr(node.child("elseExpr"), _P_LOWEST)
            }
            NodeKind.Coalesce => {
                self.expr(node.child("left"), _P_COALESCE + 1)
                self.emit(" ?? ")
                self.expr(node.child("right"), _P_COALESCE)
            }
            NodeKind.Call => { self.call(node) }
            NodeKind.Index => {
                self.receiver(node.child("object"))
                self.emit("[")
                # `a[x, y]` parses to a tuple index, and the two spell the same
                # tree — but `Dict[K, V]` is how a generic instantiation is
                # written, so print the bare list and not `Dict[(K, V)]`. A
                # one-element tuple keeps its brackets: `a[x,]` is not `a[x]`.
                let subscript = node.child("index")
                if subscript is not null and
                    subscript.kind == NodeKind.TupleLit and
                    subscript.child_list("items").len() > 1 {
                    self.sequence(subscript.child_list("items"), "", "", false)
                } else {
                    self.expr(subscript, _P_LOWEST)
                }
                self.emit("]")
            }
            NodeKind.Slice => { self.slice(node) }
            NodeKind.Member => {
                self.receiver(node.child("object"))
                self.emit("." + node.text("name"))
            }
            NodeKind.OptMember => {
                self.receiver(node.child("object"))
                self.emit("?." + node.text("name"))
            }
            NodeKind.Lambda => { self.lambda(node) }
            NodeKind.AnonFn => { self.function(node, "fn") }
            NodeKind.Comprehension => { self.comprehension(node) }
            NodeKind.Range => {
                self.expr(node.child("start"), _P_RANGE + 1)
                self.emit(node.flag("inclusive") ? "..=" : "..")
                self.expr(node.child("stop"), _P_RANGE + 1)
            }
            NodeKind.IfExpr => { self.conditional(node) }
            NodeKind.Match | NodeKind.MatchExpr => { self.match_node(node) }
            # The checker inserts casts; they have no syntax of their own.
            NodeKind.Cast => { self.bare_expr(node.require("operand")) }
            NodeKind.Yield => {
                self.emit("yield")
                let operand = node.child("operand")
                if operand is not null {
                    self.emit(" ")
                    self.expr(operand, _P_LOWEST)
                }
            }
            NodeKind.Await => {
                self.emit("await ")
                self.expr(node.child("operand"), _P_LOWEST)
            }
            NodeKind.ThrowExpr => {
                self.emit("throw ")
                self.expr(node.child("operand"), _P_LOWEST)
            }
            _ => {
                if is_pattern(node.kind) {
                    self.pattern(node)
                } else {
                    throw ValueError(f"{kind_name(node.kind)} is not an expression")
                }
            }
        }
    }

    fn sequence(self, items: list[Node], open: str, close: str, tuple_comma: bool) -> void {
        self.emit(open)
        for (index, item) in items.enumerate() {
            if index > 0 { self.emit(", ") }
            self.expr(item, _P_LOWEST)
        }
        # `(x)` is just `x`; a one-element tuple needs its trailing comma.
        if tuple_comma and items.len() == 1 { self.emit(",") }
        self.emit(close)
    }

    fn set_lit(self, node: Node) -> void {
        let items = node.child_list("items")
        # `{}` is the empty dict, so the empty set has to be spelled as a call.
        if items.len() == 0 {
            self.emit("set()")
            return
        }
        self.sequence(items, "{", "}", false)
    }

    fn dict_lit(self, node: Node) -> void {
        let keys = node.child_list("keys")
        let values = node.child_list("values")
        self.emit("{")
        for (index, key) in keys.enumerate() {
            if index > 0 { self.emit(", ") }
            self.expr(key, _P_LOWEST)
            self.emit(": ")
            if index < values.len() { self.expr(values[index], _P_LOWEST) }
        }
        self.emit("}")
    }

    fn unary(self, node: Node) -> void {
        let op = node.op()
        let operand = node.child("operand")
        if op == OpKind.Not {
            self.emit("not ")
            self.expr(operand, _P_COMPARE)
            return
        }
        self.emit(op_text(op))
        # `- -x`: without the space the two signs would read as one token. A
        # sign-leading *literal* needs no space here — `expr` brackets it.
        if _leads_with_sign(operand) { self.emit(" ") }
        self.expr(operand, _P_UNARY)
    }

    fn binary(self, node: Node) -> void {
        let op = node.op()
        let level = _OP_PRECEDENCE[op]
        if op == OpKind.Pow {
            # `**` is the one right-associative binary operator, and its left
            # operand may only be a postfix expression.
            self.expr(node.child("left"), _P_POSTFIX)
            self.emit(" ** ")
            self.expr(node.child("right"), _P_UNARY)
            return
        }
        # Comparison is chained, not left-associative (§4.2): printing the left
        # operand of `(a > b) != c` without its brackets would re-parse as the
        # three-operand chain `a > b and b != c`, a different program. Every
        # other level is left-associative and needs no bracket on the left.
        let left_level = if level == _P_COMPARE { level + 1 } else { level }
        self.expr(node.child("left"), left_level)
        self.emit(" " + op_text(op) + " ")
        self.expr(node.child("right"), level + 1)
    }

    fn compare_chain(self, node: Node) -> void {
        let operands = node.child_list("operands")
        let ops = node.get("ops")
        for (index, operand) in operands.enumerate() {
            if index > 0 { self.emit(" " + op_text(ops[index - 1]) + " ") }
            self.expr(operand, _P_COMPARE + 1)
        }
    }

    fn call(self, node: Node) -> void {
        self.receiver(node.child("callee"))
        self.emit("(")
        for (index, current) in node.records("args").enumerate() {
            if index > 0 { self.emit(", ") }
            if _flag(current, "isSpread") { self.emit("...") }
            let name = _text_at(current, "name")
            if name is not null { self.emit(name + ": ") }
            self.expr(_node_at(current, "value"), _P_LOWEST)
        }
        self.emit(")")
    }

    fn slice(self, node: Node) -> void {
        self.receiver(node.child("object"))
        self.emit("[")
        self.expr(node.child("start"), _P_LOWEST)
        self.emit(":")
        self.expr(node.child("stop"), _P_LOWEST)
        let step = node.child("step")
        if step is not null {
            self.emit(":")
            self.expr(step, _P_LOWEST)
        }
        self.emit("]")
    }

    fn lambda(self, node: Node) -> void {
        self.emit("|")
        for (index, current) in node.records("params").enumerate() {
            if index > 0 { self.emit(", ") }
            self.emit(_text_at(current, "name") ?? "_")
        }
        self.emit("| ")
        self.expr(node.child("body"), _P_LOWEST)
    }

    fn comprehension(self, node: Node) -> void {
        let kind = node.get("compKind")
        var open = "["
        var close = "]"
        if kind == CompKind.Dict or kind == CompKind.Set {
            open = "{"
            close = "}"
        } elif kind == CompKind.Generator {
            open = "("
            close = ")"
        }
        self.emit(open)
        if kind == CompKind.Dict {
            self.expr(node.child("keyExpr"), _P_LOWEST)
            self.emit(": ")
        }
        self.expr(node.child("element"), _P_LOWEST)
        for clause in node.records("clauses") {
            self.emit(" for ")
            self.pattern(_node_at(clause, "pattern"))
            self.emit(" in ")
            # A ternary iterable would swallow the clause's own `if`.
            self.expr(_node_at(clause, "iterable"), _P_COALESCE)
            for condition in clause.get("conditions", []) {
                self.emit(" if ")
                self.expr(condition, _P_COALESCE)
            }
        }
        self.emit(close)
    }

    fn fstring(self, node: Node) -> void {
        self.emit("f\"")
        for part in node.child_list("parts") {
            if part.kind == NodeKind.StrLit {
                self.emit(_fstring_chunk(part.opt_text("chars") ?? ""))
                continue
            }
            let spec = _format_spec(part)
            self.emit("{")
            if spec is null {
                self.expr(part, _P_LOWEST)
            } else {
                self.expr(part.require("callee").child("object"), _P_LOWEST)
                self.emit(":" + spec)
            }
            self.emit("}")
        }
        self.emit("\"")
    }

    fn pattern(self, node: Node?) -> void {
        if node is null {
            self.emit("_")
            return
        }
        match node.kind {
            NodeKind.PatWildcard => { self.emit("_") }
            NodeKind.PatBind => {
                self.emit(node.text("name"))
                let declared = node.type_field("type")
                if declared is not null { self.emit(": " + declared.to_source()) }
            }
            NodeKind.PatLiteral => { self.expr(node.child("value"), _P_UNARY) }
            NodeKind.PatRange => {
                self.expr(node.child("lo"), _P_RANGE + 1)
                self.emit(node.flag("inclusive") ? "..=" : "..")
                self.expr(node.child("hi"), _P_RANGE + 1)
            }
            NodeKind.PatTuple => { self.sequence_pattern(node, "(", ")") }
            NodeKind.PatList => { self.sequence_pattern(node, "[", "]") }
            NodeKind.PatOr => {
                for (index, element) in node.child_list("elems").enumerate() {
                    if index > 0 { self.emit(" | ") }
                    self.pattern(element)
                }
            }
            NodeKind.PatClass => { self.class_pattern(node) }
            NodeKind.PatEnum => { self.enum_pattern(node) }
            _ => { self.expr(node, _P_LOWEST) }
        }
    }

    fn sequence_pattern(self, node: Node, open: str, close: str) -> void {
        let elements = node.child_list("elems")
        let rest = node.number("restIndex")
        self.emit(open)
        for (index, element) in elements.enumerate() {
            if index > 0 { self.emit(", ") }
            if index == rest { self.emit("...") }
            self.pattern(element)
        }
        if open == "(" and elements.len() == 1 and rest < 0 { self.emit(",") }
        self.emit(close)
    }

    fn class_pattern(self, node: Node) -> void {
        self.emit(node.text("typeName"))
        let names = node.get("fieldNames")
        let subs = node.child_list("subPatterns")
        # `Some(x)` encodes as a fieldNames array of nulls, one per sub-pattern,
        # so emptiness is not the test for the positional form.
        if not _has_field_names(names) {
            self.emit("(")
            for (index, sub) in subs.enumerate() {
                if index > 0 { self.emit(", ") }
                self.pattern(sub)
            }
            self.emit(")")
            return
        }
        self.emit("{")
        for (index, sub) in subs.enumerate() {
            if index > 0 { self.emit(", ") }
            let name = _at_or_null(names, index)
            if name is null {
                self.pattern(sub)
                continue
            }
            # `Point{x}` is `Point{x: x}` written the short way.
            if _is_plain_bind(sub, name) {
                self.emit(name)
            } else {
                self.emit(name + ": ")
                self.pattern(sub)
            }
        }
        self.emit("}")
    }

    fn enum_pattern(self, node: Node) -> void {
        self.emit(node.text("typeName") + "." + node.text("variantName"))
        let subs = node.child_list("subPatterns")
        if subs.len() == 0 { return }
        self.emit("(")
        for (index, sub) in subs.enumerate() {
            if index > 0 { self.emit(", ") }
            self.pattern(sub)
        }
        self.emit(")")
    }
}

fn _is_block_decl(node: Node) -> bool {
    return node.kind in _BLOCK_DECLARATIONS
}

# Whether a class pattern named its fields, `Point{x: a}`, rather than matching
# them positionally, `Point(a)`.
fn _has_field_names(names: list[str?]) -> bool {
    for name in names {
        if name is not null { return true }
    }
    return false
}

fn _at_or_null(names: list[str?], index: int) -> str? {
    if index >= names.len() { return null }
    return names[index]
}

fn _is_plain_bind(node: Node?, name: str) -> bool {
    if node is null or node.kind != NodeKind.PatBind { return false }
    if node.type_field("type") is not null { return false }
    return node.text("name") == name
}

fn _type_source(type_expr: TypeExpr?) -> str {
    if type_expr is null { return "any" }
    return type_expr.to_source()
}

# True when `node` renders as a `-` or `+` applied to something, so that a
# prefix `-` or `+` in front of it needs a space to stay a token of its own.
fn _leads_with_sign(node: Node?) -> bool {
    if node is null or node.kind != NodeKind.Unary { return false }
    return node.op() == OpKind.Neg or node.op() == OpKind.Pos
}

# A literal whose own text opens with a sign. `expr` brackets one wherever it
# appears, because a space in front is not enough: the sign has to stay part of
# the token. `freeze_literals` rewrites a literal into an `Ident` holding its
# spelling, so that shape counts too.
fn _is_signed_literal(node: Node?) -> bool {
    if node is null { return false }
    match node.kind {
        NodeKind.Ident => {
            let name = node.opt_text("name") ?? ""
            return name.starts_with("-") or name.starts_with("+")
        }
        NodeKind.IntLit => { return node.number("intLit") < 0 }
        NodeKind.FloatLit => { return node.get("floatLit") < 0.0 }
        _ => { return false }
    }
}

# A float literal has to keep its point: `1` would parse back as an int.
fn _float_source(value: float) -> str {
    let text = repr(value)
    if "." in text or "e" in text or "E" in text { return text }
    return text + ".0"
}

# The literal chunks of an f-string carry their braces doubled.
fn _fstring_chunk(text: str) -> str {
    let out = StringBuilder()
    for character in text {
        if character == "{" {
            out.push("{{")
        } elif character == "}" {
            out.push("}}")
        } else {
            out.push(escape(character))
        }
    }
    return out.build()
}

# The parser rewrites `{x:.2f}` into `x.__format__(".2f")`; recover the spec so
# that the hole prints the way it was written.
fn _format_spec(node: Node) -> str? {
    if node.kind != NodeKind.Call { return null }
    let callee = node.child("callee")
    if callee is null or callee.kind != NodeKind.Member { return null }
    if callee.text("name") != "__format__" { return null }
    let args = node.records("args")
    if args.len() != 1 { return null }
    let spec = _node_at(args[0], "value")
    if spec is null or spec.kind != NodeKind.StrLit { return null }
    if _text_at(args[0], "name") is not null { return null }
    return spec.opt_text("chars")
}
