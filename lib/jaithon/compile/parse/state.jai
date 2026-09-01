#: The `Parser` class -- the token cursor, the flags a trial parse saves and
#: restores, the four entry points a caller reaches -- and every helper that
#: moves or inspects the cursor without parsing a production: token access,
#: span construction, the diagnostic constructors, expectation and recovery,
#: statement termination and synchronisation, and the reserved-name checks.
#:
#: Those helpers stayed methods on purpose. They are called from all six
#: production files, and a method call is what the moved code already wrote, so
#: keeping them here turned `self._check(...)` into `p._check(...)` and left
#: every call shape alone.
#:
#: The seams. Jaithon rejects a circular import and the productions are one
#: knot of mutual recursion, so `parse/` is ordered `type_expr -> primary ->
#: expr -> pattern -> stmt -> decl` and each file imports only the files below
#: it. Seven calls run the other way, and each is a hook here behind a one-line
#: function any file may call. The indirect call sits inside that one line
#: rather than inside the hot body that made it, so a caller keeps whatever the
#: JIT was already doing for it. `wire.jai` fills the hooks in as it is
#: imported, which is why a `Parser` built by hand -- `repl.jai` and `mod.jai`
#: both do -- needs no wiring call of its own, and why an unwired hook is a
#: thrown error naming itself rather than a wrong tree.
#:
#: What is NOT here: every production. They are free functions in the six
#: sibling files and reach this class through `p`.
from jaithon.ast import Node, NodeKind, Span, Visibility
from jaithon.compile.diag import Code, Diagnostic, DiagnosticBag
from jaithon.compile.lexer import Lexer
from jaithon.compile.token import Token, TokenKind, kind_text
from jaithon.compile.parse.node_build import _join, _node
from jaithon.compile.parse.predicate import _is_reserved_name, _starts_declaration
from jaithon.compile.parse.predicate import _starts_expression_kind

#: How deeply the recursive productions may nest before `E0107` stops them.
pub const MAX_RECURSION: int = 200

const _REPL_AS_STATEMENT = 0
const _REPL_AS_EXPRESSION = 1
const _REPL_NEED_MORE = 2

#: The parser.
pub class Parser {
    #: The token stream being read.
    pub let tokens: list[Token]
    #: Where diagnostics go; shared with the lexer that produced the tokens.
    pub var diagnostics: DiagnosticBag
    #: True once any syntax error has been reported.
    pub var had_error: bool

    pub let lexer: Lexer
    pub var current: int
    pub var panic_mode: bool
    pub var expr_depth: int
    pub var yield_count: int

    pub var trial_active: bool
    var trial_failed: bool
    var trial_error_tok: int

    #: Tokenize `lexer` and prepare to parse the tokens it produced.
    pub fn init(self, lexer: Lexer) {
        self.lexer = lexer
        self.tokens = lexer.tokenize()
        self.diagnostics = lexer.diagnostics
        self.had_error = false
        self.current = 0
        self.panic_mode = false
        self.expr_depth = 0
        self.yield_count = 0
        self.trial_active = false
        self.trial_failed = false
        self.trial_error_tok = 0
        self._skip_error_tokens()
    }

    #: Parse a whole module, or return null when the file had any error.
    #:
    #: ```
    #: let parser = Parser(Lexer("fn main() -> int { return 0 }\n"))
    #: parser.parse_program()?.kind == NodeKind.Program     # true
    #: parser.had_error                                     # false
    #: ```
    pub fn parse_program(self) -> Node? {
        self.yield_count = 0
        self._skip_error_tokens()
        self._skip_newlines()

        var stmts: list[Node] = []
        while not self._check(TokenKind.Eof) {
            let before = self.current
            let declaration = parse_declaration(self)
            if declaration != null { stmts.push(declaration) }
            if self.panic_mode { self._synchronize() }
            if self.current == before { self._advance() }  # guarantee progress
            self._skip_newlines()
        }

        let eof = self.tokens[self.tokens.len() - 1].span
        let program = _node(NodeKind.Program, Span(eof.file, 0, eof.end), {"stmts": stmts})
        if self.had_error { return null }
        return program
    }

    #: Parse a single expression and nothing else, for `--eval` and the REPL.
    pub fn parse_expression_only(self) -> Node? {
        self.yield_count = 0
        self._skip_error_tokens()
        self._skip_newlines()
        let expr = parse_expression(self)
        self._skip_newlines()
        if not self._check(TokenKind.Eof) and not self.panic_mode {
            self._error_at_cur(
                Code.E0108_TRAILING_TOKENS,
                f"unexpected {self._describe(self._peek())} after the expression"
            )
        }
        if self.had_error { return null }
        return expr
    }

    # --- the REPL line ---------------------------------------------------

    #: Read one prompt input, choosing between the two readings a prompt has.
    #:
    #: ```
    #: Parser(Lexer("1 + 1\n")).parse_repl_line()[1]     # false
    #: Parser(Lexer("1 +\n")).parse_repl_line()[1]       # true
    #: ```
    pub fn parse_repl_line(self) -> tuple[Node?, bool] {
        self.yield_count = 0
        self._skip_error_tokens()
        self._skip_newlines()
        if self._check(TokenKind.Eof) { return (null, false) }

        let reading = self._choose_repl_reading()
        if reading == _REPL_NEED_MORE { return (null, true) }
        if reading == _REPL_AS_EXPRESSION {
            return (self._read_repl_expression(), false)
        }
        return (self._read_repl_statement(), false)
    }

    fn _choose_repl_reading(self) -> int {
        let (statement_ok, statement_ran_out, statement_tok) = self._repl_trial(false)
        if statement_ok { return _REPL_AS_STATEMENT }
        if statement_ran_out { return _REPL_NEED_MORE }
        if not self._check(TokenKind.LBrace) { return _REPL_AS_STATEMENT }

        let (expression_ok, expression_ran_out, expression_tok) = self._repl_trial(true)
        if expression_ok { return _REPL_AS_EXPRESSION }
        if expression_ran_out { return _REPL_NEED_MORE }

        if expression_tok > statement_tok { return _REPL_AS_EXPRESSION }
        return _REPL_AS_STATEMENT
    }

    fn _repl_trial(self, as_expression: bool) -> tuple[bool, bool, int] {
        let mark_current = self.current
        let mark_expr_depth = self.expr_depth
        let mark_yield_count = self.yield_count
        let mark_had_error = self.had_error
        let mark_panic_mode = self.panic_mode

        self.trial_active = true
        self.trial_failed = false
        self.trial_error_tok = self.current

        if as_expression {
            self._read_repl_expression()
        } else {
            self._read_repl_statement()
        }

        let ok = not self.trial_failed and not self.had_error
        let error_tok = self.trial_error_tok
        let ran_out = self.trial_failed and self._token_at(error_tok).kind == TokenKind.Eof
        self.trial_active = false

        self.current = mark_current
        self.expr_depth = mark_expr_depth
        self.yield_count = mark_yield_count
        self.had_error = mark_had_error
        self.panic_mode = mark_panic_mode
        return (ok, ran_out, error_tok)
    }

    fn _read_repl_statement(self) -> Node? {
        let node = parse_declaration(self)
        if self.panic_mode { self._synchronize() }
        self._skip_newlines()
        self._expect_repl_end()
        if self.had_error { return null }
        return node
    }

    fn _read_repl_expression(self) -> Node? {
        let start_index = self.current
        let expr = parse_expression(self)
        if self.panic_mode { self._synchronize() }
        self._skip_newlines()
        self._expect_repl_end()
        if self.had_error { return null }
        return _node(NodeKind.ExprStmt, self._span_since(start_index), {"expr": expr})
    }

    fn _expect_repl_end(self) -> void {
        if self._check(TokenKind.Eof) or self.had_error { return }
        self._error_at_cur(
            Code.E0108_TRAILING_TOKENS,
            f"unexpected {self._describe(self._peek())} after the statement"
        )
    }

    # --- token access ----------------------------------------------------
    pub fn _token_at(self, index: int) -> Token {
        if index < 0 or index >= self.tokens.len() {
            return self.tokens[self.tokens.len() - 1]  # always the Eof token
        }
        return self.tokens[index]
    }

    pub fn _peek(self) -> Token {
        return self._token_at(self.current)
    }

    pub fn _kind(self) -> TokenKind {
        return self._peek().kind
    }

    pub fn _kind_at(self, offset: int) -> TokenKind {
        return self._token_at(self.current + offset).kind
    }

    pub fn _check(self, kind: TokenKind) -> bool {
        return self._kind() == kind
    }

    pub fn _advance(self) -> Token {
        let token = self._peek()
        if token.kind != TokenKind.Eof {
            self.current += 1
            self._skip_error_tokens()
        }
        return token
    }

    pub fn _match(self, kind: TokenKind) -> bool {
        if not self._check(kind) { return false }
        self._advance()
        return true
    }

    pub fn _skip_newlines(self) -> void {
        while self._check(TokenKind.Newline) { self._advance() }
    }

    fn _skip_error_tokens(self) -> void {
        while self._token_at(self.current).kind == TokenKind.Error {
            self.had_error = true
            self.current += 1
        }
    }

    pub fn _cur_span(self) -> Span {
        return self._peek().span
    }

    pub fn _span_since(self, start_index: int) -> Span {
        var end_index = self.current - 1
        while end_index > start_index and self._token_at(end_index).kind == TokenKind.Newline {
            end_index -= 1
        }
        if end_index < start_index { end_index = start_index }
        return _join(self._token_at(start_index).span, self._token_at(end_index).span)
    }

    pub fn _ident_text(self, offset: int, text: str) -> bool {
        let token = self._token_at(self.current + offset)
        return token.kind == TokenKind.Ident and token.text == text
    }

    pub fn _label_name(self, token: Token) -> str {
        if token.text.len() > 0 and token.text[0] == "'" { return token.text[1:token.text.len()] }
        return token.text
    }

    # --- diagnostics -----------------------------------------------------
    pub fn _describe(self, token: Token) -> str {
        if token.kind == TokenKind.Eof { return "end of file" }
        if token.kind == TokenKind.Newline { return "a newline" }
        if token.text.len() == 0 { return f"`{kind_text(token.kind)}`" }
        if token.text.len() > 32 { return f"`{token.text[0:32]}`" }
        return f"`{token.text}`"
    }

    pub fn _error_at(
        self,
        span: Span,
        code: str,
        message: str,
        help: str? = null,
        note: str? = null
    ) -> Diagnostic? {
        if self.panic_mode { return null }
        self.panic_mode = true
        self.had_error = true
        if self.trial_active { return self._trial_note_error() }
        return self.diagnostics.error(code, span, message, help: help, note: note)
    }

    fn _trial_note_error(self) -> Diagnostic? {
        if not self.trial_failed {
            self.trial_failed = true
            self.trial_error_tok = self.current
        }
        return null
    }

    pub fn _error_at_cur(self, code: str, message: str, help: str? = null) -> Diagnostic? {
        return self._error_at(self._cur_span(), code, message, help: help)
    }

    pub fn _error_recovered(
        self,
        span: Span,
        code: str,
        message: str,
        help: str? = null
    ) -> Diagnostic? {
        if self.panic_mode { return null }
        self.had_error = true
        if self.trial_active { return self._trial_note_error() }
        return self.diagnostics.error(code, span, message, help: help)
    }

    pub fn _expect(self, kind: TokenKind, what: str) -> bool {
        if self._check(kind) {
            self._advance()
            return true
        }
        self._error_at(
            self._cur_span(),
            Code.E0101_EXPECTED_TOKEN,
            f"expected {what}, found {self._describe(self._peek())}"
        )
        return false
    }

    pub fn _unclosed(self, open_index: int, close_text: str, what: str) -> void {
        let found = self._describe(self._peek())
        let open_span = self._token_at(open_index).span
        let diag = self._error_at(
            open_span,
            Code.E0107_UNCLOSED_DELIMITER,
            f"expected `{close_text}` to close this {what}, found {found}"
        )
        if diag != null {
            diag.label(open_span, f"unclosed {what} opened here")
            if not self._check(TokenKind.Eof) {
                diag.label(self._cur_span(), f"{found} reached before the `{close_text}`")
            }
            diag.help = f"add the matching `{close_text}`"
        }
    }

    pub fn _expect_ident_name(self, what: str) -> str? {
        if self._check(TokenKind.Ident) { return self._advance().text }
        if self._check(TokenKind.Underscore) {
            self._advance()
            return "_"
        }
        self._error_at(
            self._cur_span(),
            Code.E0106_EXPECTED_IDENTIFIER,
            f"expected {what}, found {self._describe(self._peek())}"
        )
        return null
    }

    pub fn _error_expr(self, span: Span) -> Node {
        return _node(NodeKind.NullLit, span)
    }

    pub fn _enter_recursion(self) -> bool {
        self.expr_depth += 1
        if self.expr_depth > MAX_RECURSION {
            self._error_at_cur(
                Code.E0107_UNCLOSED_DELIMITER,
                f"this construct nests more than {MAX_RECURSION} levels deep"
            )
            return false
        }
        return true
    }

    # --- statement termination and synchronisation -----------------------
    pub fn _synchronize(self) -> void {
        self.panic_mode = false
        if self._check(TokenKind.Eof) or self._check(TokenKind.RBrace) { return }

        var depth = 0
        self._advance()  # always make progress past the offending token

        while not self._check(TokenKind.Eof) {
            let kind = self._kind()
            if depth == 0 {
                if kind == TokenKind.Newline {
                    self._skip_newlines()
                    return
                }
                if kind == TokenKind.RBrace { return }
                if _starts_declaration(kind) { return }
            }
            if kind == TokenKind.LBrace or kind == TokenKind.LParen or kind == TokenKind.LBracket {
                depth += 1
            } elif kind == TokenKind.RBrace or
                kind == TokenKind.RParen or
                kind == TokenKind.RBracket {
                if depth == 0 { return }
                depth -= 1
            }
            self._advance()
        }
    }

    pub fn _end_statement(self) -> void {
        if self._check(TokenKind.Semicolon) {
            self._error_recovered(
                self._cur_span(),
                Code.E0100_UNEXPECTED_TOKEN,
                "Jaithon has no statement separators",
                help: "delete the `;` — a newline ends a statement"
            )
            self._advance()
        }
        if self.panic_mode { return }
        if self._check(TokenKind.Newline) {
            self._skip_newlines()
            return
        }
        if self._check(TokenKind.RBrace) or self._check(TokenKind.Eof) { return }
        self._error_at_cur(
            Code.E0101_EXPECTED_TOKEN,
            "expected a newline after this statement, " + f"found {self._describe(self._peek())}"
        )
    }

    pub fn _starts_expression(self) -> bool {
        return _starts_expression_kind(self._kind())
    }

    pub fn _check_reserved_name(self, name: str?, span: Span) -> void {
        if name == null or not _is_reserved_name(name) { return }
        self._error_recovered(
            span,
            Code.E0114_RESERVED_IDENTIFIER,
            f"`{name}` is reserved for the implementation",
            help: "names of the form `__x__` may not be declared here"
        )
    }

    pub fn _check_reserved_type_name(self, name: str?, span: Span) -> void {
        if name != "Self" { return }
        self._error_recovered(
            span,
            Code.E0114_RESERVED_IDENTIFIER,
            "`Self` is reserved: it names the implementing type",
            help: "`Self` is already in scope inside every class, trait and enum body"
        )
    }

    pub fn _check_duplicate_param(
        self,
        params: list[dict[str, any]],
        name: str,
        span: Span
    ) -> void {
        for existing in params {
            if existing["name"] != name { continue }
            let diag = self._error_recovered(
                span,
                Code.E0111_DUPLICATE_PARAMETER,
                f"duplicate parameter `{name}`"
            )
            if diag != null { diag.label(existing["span"], "first declared here") }
            return
        }
    }
}

# --- the seams onto the production files -------------------------------
#: Each default throws under its own name: a parser has no useful degraded
#: mode, and answering a wrong tree would be found much later than a throw.
fn _unwired_parse_declaration(p: Parser) -> Node? {
    throw ValueError("the parser seam `parse_declaration` was never wired")
}

fn _unwired_parse_expression(p: Parser) -> Node {
    throw ValueError("the parser seam `parse_expression` was never wired")
}

fn _unwired_parse_pattern(p: Parser) -> Node {
    throw ValueError("the parser seam `parse_pattern` was never wired")
}

fn _unwired_parse_fn_rest(
    p: Parser,
    visibility: Visibility,
    is_static: bool,
    start_index: int,
    named: bool,
    body_optional: bool
) -> Node? {
    throw ValueError("the parser seam `parse_fn_rest` was never wired")
}

fn _unwired_parse_if(p: Parser, as_expr: bool) -> Node {
    throw ValueError("the parser seam `parse_if` was never wired")
}

fn _unwired_parse_match(p: Parser, as_expr: bool) -> Node {
    throw ValueError("the parser seam `parse_match` was never wired")
}

fn _unwired_parse_unary(p: Parser) -> Node {
    throw ValueError("the parser seam `parse_unary` was never wired")
}

var _hook_parse_declaration: fn(Parser) -> Node? = _unwired_parse_declaration
var _hook_parse_expression: fn(Parser) -> Node = _unwired_parse_expression
var _hook_parse_pattern: fn(Parser) -> Node = _unwired_parse_pattern
var _hook_parse_fn_rest: fn(Parser, Visibility, bool, int, bool, bool) -> Node? = _unwired_parse_fn_rest
var _hook_parse_if: fn(Parser, bool) -> Node = _unwired_parse_if
var _hook_parse_match: fn(Parser, bool) -> Node = _unwired_parse_match
var _hook_parse_unary: fn(Parser) -> Node = _unwired_parse_unary

#: `decl.jai`'s `_parse_declaration`, for this file and for `stmt.jai`.
pub fn parse_declaration(p: Parser) -> Node? {
    return _hook_parse_declaration(p)
}

#: `expr.jai`'s `_parse_expression`, for this file and for `primary.jai`.
pub fn parse_expression(p: Parser) -> Node {
    return _hook_parse_expression(p)
}

#: `pattern.jai`'s `_parse_pattern`, for a comprehension clause in `primary.jai`.
pub fn parse_pattern(p: Parser) -> Node {
    return _hook_parse_pattern(p)
}

#: `decl.jai`'s `_parse_fn_rest`, for an anonymous `fn` in `primary.jai` and
#: for the legacy call statement in `stmt.jai`.
pub fn parse_fn_rest(
    p: Parser,
    visibility: Visibility,
    is_static: bool,
    start_index: int,
    named: bool,
    body_optional: bool
) -> Node? {
    return _hook_parse_fn_rest(p, visibility, is_static, start_index, named, body_optional)
}

#: `stmt.jai`'s `_parse_if`, for `if` in expression position in `primary.jai`.
pub fn parse_if(p: Parser, as_expr: bool) -> Node {
    return _hook_parse_if(p, as_expr)
}

#: `stmt.jai`'s `_parse_match`, for `match` in expression position in `primary.jai`.
pub fn parse_match(p: Parser, as_expr: bool) -> Node {
    return _hook_parse_match(p, as_expr)
}

#: `expr.jai`'s `_parse_unary`, for the operand of `await` in `primary.jai`.
pub fn parse_unary(p: Parser) -> Node {
    return _hook_parse_unary(p)
}

#: Point every seam at the production that implements it. `wire.jai` is the
#: only caller, and it calls this as it is imported.
pub fn install_hooks(
    declaration: fn(Parser) -> Node?,
    expression: fn(Parser) -> Node,
    pattern: fn(Parser) -> Node,
    fn_rest: fn(Parser, Visibility, bool, int, bool, bool) -> Node?,
    if_form: fn(Parser, bool) -> Node,
    match_form: fn(Parser, bool) -> Node,
    unary: fn(Parser) -> Node
) -> void {
    _hook_parse_declaration = declaration
    _hook_parse_expression = expression
    _hook_parse_pattern = pattern
    _hook_parse_fn_rest = fn_rest
    _hook_parse_if = if_form
    _hook_parse_match = match_form
    _hook_parse_unary = unary
}
