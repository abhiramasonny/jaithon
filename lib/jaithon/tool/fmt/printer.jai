#: The printer: one method per syntax form, and the one-line-block and
#: line-breaking decisions that go with them.
from jaithon.ast import DeclKind, Node, NodeKind
from jaithon.ast import op_text
from jaithon.ast_unparse import to_source
from std.str import repeat
from .comment import Comment, _Trivia
from .layout import LINE_WIDTH, _Member, _broken_chain, _by_position
from .layout import _class_members, _collapse, _has_block, _is_declaration
from .layout import _label_prefix, _label_suffix, _method_members, _type_source
from .layout import _visibility
from .source import _Source, _dedent_block
from .text import _INDENT, _indent_block, _strip_end, _triple_quotes

class _Line {
    pub let text: str
    pub var comment: str?
    pub let depth: int

    pub fn init(self, text: str, depth: int) {
        self.text = text
        self.comment = null
        self.depth = depth
    }
}

pub class _Printer {
    pub let src: _Source
    pub let trivia: _Trivia
    pub let out: list[_Line]
    pub var depth: int
    pub var last_line: int

    pub fn init(self, src: _Source, trivia: _Trivia) {
        self.src = src
        self.trivia = trivia
        self.out = []
        self.depth = 0
        self.last_line = 0
    }

    # --- output ------------------------------------------------------------
    pub fn emit(self, text: str) -> void {
        let prefix = repeat(_INDENT, self.depth)
        var inside_literal = false
        for piece in text.split("\n") {
            if inside_literal {
                #: A line inside a triple-quoted string is the program's DATA,
                #: not its layout. Indenting it changes what the string holds:
                #: `"""alpha\nbeta\n"""` came back four characters longer with
                #: `beta` no longer at the start of its line, and nothing said
                #: so. Written through with no prefix and no trailing trim.
                self.out.push(_Line(piece, 0))
            } else {
                let trimmed = _strip_end(piece)
                let full = if trimmed.len() == 0 { "" } else { prefix + trimmed }
                self.out.push(_Line(full, self.depth))
            }
            if _triple_quotes(piece) % 2 == 1 { inside_literal = not inside_literal }
        }
    }

    #: A function's decorators, each on its own line ahead of its signature.
    #:
    #: They are not part of the signature and were not written at all, so
    #: formatting a file deleted every one of them -- silently, and with
    #: consequences: `@trace` is what puts a function in the traced tier and
    #: `@gpu_kernel` is what makes one a kernel. A pass over jaitensor turned
    #: its hot paths back into ordinary functions and exactly one test noticed.
    pub fn decorators(self, node: Node) -> void {
        if not node.has("decorators") { return }
        let names = node.get("decorators")
        if names is null { return }
        for name in names { self.emit(f"@{name}") }
    }

    pub fn header(self, text: str, line: int) -> void {
        self.emit(text)
        let trailing = self.trivia.take_trailing(line)
        if trailing is not null { self.attach(trailing) }
    }

    pub fn blank(self) -> void {
        if self.out.len() == 0 { return }
        let previous = self.out[self.out.len() - 1].text
        if previous.len() == 0 or previous.ends_with("{") { return }
        self.out.push(_Line("", self.depth))
    }

    pub fn gap_to(self, line: int) -> void {
        if self.last_line > 0 and line > self.last_line + 1 { self.blank() }
        self.last_line = line
    }

    pub fn attach(self, comment: Comment) -> void {
        if self.out.len() == 0 { return }
        self.out[self.out.len() - 1].comment = comment.text
    }

    pub fn column(self) -> int { return self.depth * _INDENT.len() }

    pub fn render(self) -> list[str] {
        let text: list[str] = []
        var index = 0
        while index < self.out.len() {
            if self.out[index].comment is null {
                text.push(self.out[index].text)
                index += 1
                continue
            }
            var last = index
            while last + 1 < self.out.len() and
                self.out[last + 1].comment is not null and
                self.out[last + 1].depth == self.out[index].depth {
                last += 1
            }
            var width = 0
            for position in index..=last { width = max(width, self.out[position].text.len()) }
            for position in index..=last {
                let line = self.out[position]
                let padding = repeat(" ", width - line.text.len() + 2)
                text.push(line.text + padding + (line.comment ?? ""))
            }
            index = last + 1
        }
        while text.len() > 0 and text[text.len() - 1].len() == 0 { text.pop() }
        return text
    }

    # --- statement sequences -----------------------------------------------
    pub fn statements(self, stmts: list[Node], close_line: int, top_level: bool) -> void {
        for statement in stmts {
            let start = self.src.line_of(statement.span.start)
            let leading = self.trivia.take_leading(start)
            let first = if leading.len() > 0 { leading[0].line } else { start }
            if top_level and self.out.len() > 0 and _is_declaration(statement) {
                self.blank()
                self.last_line = first
            } else {
                self.gap_to(first)
            }
            self.comments(leading)
            self.last_line = start
            let end = self.src.line_of(statement.span.end)
            if not self.statement(statement, start, end) {
                let trailing = self.trivia.take_trailing(end)
                if trailing is not null { self.attach(trailing) }
            }
            self.last_line = end
        }
        self.comments(self.trivia.take_leading(close_line))
    }

    pub fn comments(self, items: list[Comment]) -> void {
        for comment in items {
            self.gap_to(comment.line)
            self.emit(comment.text)
            self.last_line = comment.end_line
        }
    }

    pub fn statement(self, node: Node, start: int, end: int) -> bool {
        if not _has_block(node) and self.trivia.any_inside(start, end) {
            self.trivia.consume_within(start, end)
            self.emit(_dedent_block(self.src, start, end))
            return true
        }

        let simple = self.simple_statement(node)
        if simple is not null {
            self.emit(simple)
            return false
        }

        if start == end {
            let probe = _Printer(self.src, _Trivia([]))
            probe.block_statement(node)
            let collapsed = _collapse(probe.render())
            if collapsed is not null and self.column() + collapsed.len() <= LINE_WIDTH {
                self.emit(collapsed)
                return false
            }
        }
        self.block_statement(node)
        return false
    }

    pub fn simple_statement(self, node: Node) -> str? {
        let column = self.column()
        match node.kind {
            NodeKind.ExprStmt => { return self.expr(node.child("expr"), column) }
            NodeKind.VarDecl => { return self.var_decl(node, column) }
            NodeKind.Assign => {
                let target = to_source(node.require("target"))
                let operator = if node.flag("isCompound") { op_text(node.op()) + "=" } else { "=" }
                let head = target + " " + operator + " "
                return head + self.expr(node.child("value"), column + head.len())
            }
            NodeKind.Return => {
                let value = node.child("value")
                if value is null { return "return" }
                return "return " + self.expr(value, column + 7)
            }
            NodeKind.Throw => { return "throw " + self.expr(node.child("value"), column + 6) }
            NodeKind.Break => { return "break" + _label_suffix(node.opt_text("label")) }
            NodeKind.Continue => { return "continue" + _label_suffix(node.opt_text("label")) }
            NodeKind.Assert => {
                let condition = self.expr(node.child("cond"), column + 7)
                let message = node.child("message")
                if message is null { return "assert " + condition }
                return "assert " + condition + ", " + self.expr(message, column)
            }
            NodeKind.Import | NodeKind.FromImport | NodeKind.Export | NodeKind.ModuleDecl => {
                return to_source(node)
            }
            NodeKind.TypeDecl => { return self.signature(node) }
            _ => { return null }
        }
    }

    # Printed by a throwaway printer holding no comments — safe because the
    # caller has already established this construct spans one source line.
    pub fn collapse_body(self, head: str, body: Node?) -> str? {
        if body is null { return null }
        let probe = _Printer(self.src, _Trivia([]))
        probe.emit(head + " {")
        probe.body(body)
        probe.emit("}")
        return _collapse(probe.render())
    }

    pub fn block_statement(self, node: Node) -> void {
        let column = self.column()
        let line = self.src.line_of(node.span.start)
        match node.kind {
            NodeKind.Block => {
                self.header("{", line)
                self.body(node)
                self.emit("}")
            }
            NodeKind.If => { self.if_statement(node) }
            NodeKind.While => {
                self.header(
                    _label_prefix(node.opt_text("label")) +
                        "while " +
                        self.expr(node.child("cond"), column + 6) +
                        " {",
                    line
                )
                self.body(node.child("body"))
                self.emit("}")
            }
            NodeKind.Loop => {
                self.header(_label_prefix(node.opt_text("label")) + "loop {", line)
                self.body(node.child("body"))
                self.emit("}")
            }
            NodeKind.For => {
                self.header(
                    _label_prefix(node.opt_text("label")) +
                        "for " +
                        to_source(node.require("pattern")) +
                        " in " +
                        self.expr(node.child("iterable"), column) +
                        " {",
                    line
                )
                self.body(node.child("body"))
                self.emit("}")
            }
            NodeKind.Match => {
                self.header("match " + self.expr(node.child("subject"), column + 6) + " {", line)
                self.depth += 1
                self.arms(node)
                self.depth -= 1
                self.emit("}")
            }
            NodeKind.Try => { self.try_statement(node) }
            NodeKind.Defer => {
                self.header("defer {", line)
                self.body(node.child("body"))
                self.emit("}")
            }
            NodeKind.FnDecl => {
                self.decorators(node)
                self.header(self.signature(node) + " {", line)
                self.body(node.child("body"))
                self.emit("}")
            }
            NodeKind.ClassDecl => { self.type_body(node, _class_members(node)) }
            NodeKind.TraitDecl => { self.type_body(node, _method_members(node)) }
            NodeKind.EnumDecl => { self.enum_decl(node) }
            _ => {
                # An unknown shape is copied rather than lost.
                let end = self.src.line_of(node.span.end)
                self.trivia.consume_within(line, end)
                self.emit(_dedent_block(self.src, line, end))
            }
        }
    }

    pub fn body(self, block: Node?) -> void {
        if block is null { return }
        self.depth += 1
        if block.kind == NodeKind.Block {
            let saved = self.last_line
            self.last_line = self.src.line_of(block.span.start)
            self.statements(block.child_list("stmts"), self.src.line_of(block.span.end), false)
            self.last_line = saved
        } else {
            self.emit(self.expr(block, self.column()))
        }
        self.depth -= 1
    }

    pub fn if_statement(self, node: Node) -> void {
        var current = node
        var opener = "if "
        loop {
            let line = self.src.line_of(current.span.start)
            self.header(
                opener + self.expr(current.child("cond"), self.column() + opener.len()) + " {",
                line
            )
            self.body(current.child("thenBranch"))
            let alternative = current.child("elseBranch")
            if alternative is null {
                self.emit("}")
                return
            }
            if alternative.kind == NodeKind.If {
                opener = "} elif "
                current = alternative
                continue
            }
            self.header("} else {", self.src.line_of(alternative.span.start))
            self.body(alternative)
            self.emit("}")
            return
        }
    }

    pub fn try_statement(self, node: Node) -> void {
        self.header("try {", self.src.line_of(node.span.start))
        self.body(node.child("body"))
        for handler in node.records("catches") {
            let name = handler["name"] ?? "_error"
            let caught = [entry.to_source() for entry in handler["types"]]
            let clause = if caught.len() == 0 { "" } else { ": " + " | ".join(caught) }
            self.header(f"}} catch {name}{clause} {{", self.src.line_of(handler["span"].start))
            self.body(handler["body"])
        }
        let ending = node.child("finallyBlock")
        if ending is not null {
            self.header("} finally {", self.src.line_of(ending.span.start))
            self.body(ending)
        }
        self.emit("}")
    }

    pub fn arms(self, node: Node) -> void {
        self.last_line = self.src.line_of(node.span.start)
        for arm in node.records("arms") {
            let start = self.src.line_of(arm["span"].start)
            let leading = self.trivia.take_leading(start)
            self.gap_to(if leading.len() > 0 { leading[0].line } else { start })
            self.comments(leading)
            self.last_line = start
            self.emit(self.match_arm(arm))
            let end = self.src.line_of(arm["span"].end)
            let trailing = self.trivia.take_trailing(end)
            if trailing is not null { self.attach(trailing) }
            self.last_line = end
        }
        self.comments(self.trivia.take_leading(self.src.line_of(node.span.end)))
    }

    pub fn match_arm(self, arm: dict[str, any]) -> str {
        let guard = arm["guard"]
        let guard_text = if guard is null { "" } else { " if " + to_source(guard) }
        let head = to_source(arm["pattern"]) + guard_text + " => "
        let body = arm["body"]
        if body is not null and body.kind == NodeKind.Block {
            let inner = self.collapse_body("", body)
            if inner is not null and self.column() + head.len() + inner.len() <= LINE_WIDTH {
                return head + inner
            }
            return head + self.nested_block(body)
        }
        return head + self.expr(body, self.column() + head.len()) + ","
    }

    # --- declarations -------------------------------------------------------
    pub fn signature(self, node: Node) -> str {
        match node.kind {
            NodeKind.FnDecl | NodeKind.AnonFn => {
                let returns = _type_source(node.type_field("returnType"))
                let arrow = if returns.len() == 0 { "" } else { " -> " + returns }
                let statics = if node.flag("isStatic") { "static " } else { "" }
                let head = _visibility(node.visibility()) +
                    statics +
                    "fn " +
                    (node.opt_text("name") ?? "") +
                    self.generics(node.records("generics"))
                return head +
                    self.params(node.records("params"), self.column() + head.len(), arrow.len() + 2) +
                    arrow
            }
            NodeKind.ClassDecl => {
                let parent = _type_source(node.type_field("superclass"))
                let extends_clause = if parent.len() == 0 { "" } else { " extends " + parent }
                let traits = [entry.to_source() for entry in node.type_list("traits")]
                let implements = if traits.len() == 0 { "" } else { ": " + ", ".join(traits) }
                return _visibility(node.visibility()) +
                    "class " +
                    node.text("name") +
                    self.generics(node.records("generics")) +
                    extends_clause +
                    implements
            }
            NodeKind.TraitDecl => {
                let supers = [entry.to_source() for entry in node.type_list("supers")]
                let bounds = if supers.len() == 0 { "" } else { ": " + ", ".join(supers) }
                return _visibility(node.visibility()) +
                    "trait " +
                    node.text("name") +
                    self.generics(node.records("generics")) +
                    bounds
            }
            NodeKind.EnumDecl => {
                return _visibility(node.visibility()) +
                    "enum " +
                    node.text("name") +
                    self.generics(node.records("generics"))
            }
            NodeKind.TypeDecl => {
                return _visibility(node.visibility()) +
                    "type " +
                    node.text("name") +
                    self.generics(node.records("generics")) +
                    " = " +
                    _type_source(node.type_field("aliased"))
            }
            NodeKind.VarDecl => { return self.var_decl(node, self.column()) }
            _ => { return to_source(node) }
        }
    }

    pub fn var_decl(self, node: Node, column: int) -> str {
        let word = match node.get("varDeclKind") {
            DeclKind.Let => "let",
            DeclKind.Var => "var",
            DeclKind.Const => "const",
        }
        let declared = _type_source(node.type_field("declaredType"))
        let annotation = if declared.len() == 0 { "" } else { ": " + declared }
        let head = _visibility(node.visibility()) +
            word +
            " " +
            to_source(node.require("pattern")) +
            annotation
        let init = node.child("init")
        if init is null { return head }
        return head + " = " + self.expr(init, column + head.len() + 3)
    }

    pub fn type_body(self, node: Node, entries: list[_Member]) -> void {
        self.header(self.signature(node) + " {", self.src.line_of(node.span.start))
        self.depth += 1
        self.members(node, entries)
        self.depth -= 1
        self.emit("}")
    }

    pub fn variant(self, entry: dict[str, any]) -> str {
        let payload = entry["params"]
        if payload.len() == 0 { return entry["name"] }
        return entry["name"] + self.params(payload, self.column())
    }

    pub fn enum_decl(self, node: Node) -> void {
        let variants = [self.variant(entry) for entry in node.records("variants")]
        let methods = node.child_list("methods")
        let open = self.src.line_of(node.span.start)
        let close = self.src.line_of(node.span.end)
        let flat = self.signature(node) + " { " + ", ".join(variants) + " }"
        if methods.len() == 0 and
            self.column() + flat.len() <= LINE_WIDTH and
            not flat.contains("\n") and
            not self.trivia.any_within(open, close) {
            self.emit(flat)
            return
        }
        self.header(self.signature(node) + " {", open)
        self.depth += 1
        self.last_line = open
        for (index, entry) in node.records("variants").enumerate() {
            let start = self.src.line_of(entry["span"].start)
            let leading = self.trivia.take_leading(start)
            self.gap_to(if leading.len() > 0 { leading[0].line } else { start })
            self.comments(leading)
            self.last_line = start
            self.emit(variants[index] + ",")
            let end = self.src.line_of(entry["span"].end)
            let trailing = self.trivia.take_trailing(end)
            if trailing is not null { self.attach(trailing) }
            self.last_line = end
        }
        if methods.len() > 0 {
            self.members(node, _method_members(node))
        } else {
            self.comments(self.trivia.take_leading(close))
        }
        self.depth -= 1
        self.emit("}")
    }

    pub fn members(self, owner: Node, entries: list[_Member]) -> void {
        self.last_line = self.src.line_of(owner.span.start)
        for (index, member) in _by_position(entries).enumerate() {
            let start = self.src.line_of(member.start)
            let leading = self.trivia.take_leading(start)
            let first = if leading.len() > 0 { leading[0].line } else { start }
            if index > 0 and member.role != "field" {
                self.blank()
                self.last_line = first
            } else {
                self.gap_to(first)
            }
            self.comments(leading)
            self.last_line = start
            let end = self.member(member, start)
            let trailing = self.trivia.take_trailing(end)
            if trailing is not null { self.attach(trailing) }
            self.last_line = end
        }
        self.comments(self.trivia.take_leading(self.src.line_of(owner.span.end)))
    }

    pub fn member(self, member: _Member, start: int) -> int {
        if member.role == "field" {
            let field = member.node
            let keyword = if field["isLet"] { "let" } else { "var" }
            let statics = if field["isStatic"] { "static " } else { "" }
            let head = _visibility(field["visibility"]) +
                statics +
                keyword +
                " " +
                field["name"] +
                ": " +
                _type_source(field["type"])
            let fallback = field["defaultValue"]
            if fallback is null {
                self.emit(head)
            } else {
                self.emit(head + " = " + self.expr(fallback, self.column() + head.len() + 3))
            }
            return self.src.line_of(field["span"].end)
        }

        let node = member.node
        self.decorators(node)
        let head = match member.role {
            "getter" => _visibility(node.visibility()) + "get " + node.text("name") + "() -> " + _type_source(node.type_field("returnType")),
            "setter" => _visibility(node.visibility()) + "set " + node.text("name") + self.params(node.records("params"), self.column()),
            _ => self.signature(node),
        }
        let end = self.src.line_of(node.span.end)
        let body = node.child("body")
        if body is null {
            self.emit(head)
            return end
        }
        if start == end and not self.trivia.any_within(start, end) {
            let inner = self.collapse_body(head, body)
            if inner is not null and self.column() + inner.len() <= LINE_WIDTH {
                self.emit(inner)
                return end
            }
        }
        self.header(head + " {", start)
        self.body(body)
        self.emit("}")
        return end
    }

    pub fn generics(self, generics: list[dict[str, any]]) -> str {
        if generics.len() == 0 { return "" }
        let parts: list[str] = []
        for entry in generics {
            let bound = _type_source(entry["bound"])
            parts.push(entry["name"] + (if bound.len() == 0 { "" } else { ": " + bound }))
        }
        return "[" + ", ".join(parts) + "]"
    }

    pub fn params(self, params: list[dict[str, any]], column: int, reserve: int = 0) -> str {
        if params.len() == 0 { return "()" }
        let parts: list[str] = []
        for param in params {
            let marker = if param["isVariadic"] {
                "..."
            } elif param["isKwRest"] {
                "**"
            } else {
                ""
            }
            let declared = _type_source(param["type"])
            let annotation = if declared.len() == 0 { "" } else { ": " + declared }
            let fallback = param["defaultValue"]
            let given = if fallback is null { "" } else { " = " + self.expr(fallback, column) }
            parts.push(marker + param["name"] + annotation + given)
        }
        let flat = "(" + ", ".join(parts) + ")"
        if column + flat.len() + reserve <= LINE_WIDTH and not flat.contains("\n") { return flat }
        return "(\n" + ",\n".join([_indent_block(part) for part in parts]) + "\n)"
    }

    # --- expressions --------------------------------------------------------
    pub fn expr(self, node: Node?, column: int) -> str {
        if node is null { return "" }
        let flat = to_source(node)
        if flat.contains("\n") {
            let folded = _collapse(flat.split("\n"))
            if folded is not null and column + folded.len() <= LINE_WIDTH { return folded }
            return flat
        }
        if column + flat.len() <= LINE_WIDTH { return flat }
        return self.broken(node, column, flat) ?? flat
    }

    pub fn broken(self, node: Node, column: int, flat: str) -> str? {
        match node.kind {
            NodeKind.Call => {
                let args = node.records("args")
                if args.len() == 0 { return null }
                let parts: list[str] = []
                for argument in args {
                    let spread = if argument["isSpread"] { "..." } else { "" }
                    let name = argument["name"]
                    let label = if name is null { "" } else { name + ": " }
                    parts.push(spread + label + self.expr(argument["value"], column + 4))
                }
                let rendered = to_source(node.require("callee"))
                let callee = if flat.starts_with(rendered + "(") { rendered } else { "(" + rendered + ")" }
                return callee + "(\n" + ",\n".join([_indent_block(part) for part in parts]) + "\n)"
            }
            NodeKind.ListLit => { return self.spread_items(node, "[", "]", column) }
            NodeKind.SetLit => { return self.spread_items(node, "{", "}", column) }
            NodeKind.TupleLit => { return self.spread_items(node, "(", ")", column) }
            NodeKind.DictLit => {
                let keys = node.child_list("keys")
                let values = node.child_list("values")
                if keys.len() == 0 { return null }
                let parts: list[str] = []
                for index in 0..keys.len() {
                    parts.push(self.expr(keys[index], column + 4) + ": " + self.expr(values[index], column + 4))
                }
                return "{\n" + ",\n".join([_indent_block(part) for part in parts]) + ",\n}"
            }
            NodeKind.Binary | NodeKind.Logical => { return _broken_chain(node, flat) }
            _ => { return null }
        }
    }

    pub fn spread_items(self, node: Node, open: str, close: str, column: int) -> str? {
        let items = node.child_list("items")
        if items.len() == 0 { return null }
        if close == ")" and items.len() == 1 { return null }
        let parts = [self.expr(item, column + 4) for item in items]
        let comma = if close == ")" { "" } else { "," }
        return open +
            "\n" +
            ",\n".join([_indent_block(part) for part in parts]) +
            comma +
            "\n" +
            close
    }

    pub fn nested_block(self, block: Node) -> str {
        let inner = _Printer(self.src, self.trivia)
        inner.depth = 1
        inner.last_line = self.src.line_of(block.span.start)
        inner.statements(block.child_list("stmts"), self.src.line_of(block.span.end), false)
        let lines = inner.render()
        if lines.len() == 0 { return "{ }" }
        return "{\n" + "\n".join(lines) + "\n}"
    }
}
