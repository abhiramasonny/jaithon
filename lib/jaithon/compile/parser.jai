#: The Jaithon parser: tokens in, a `jaithon.ast` tree out.
#:
#: Recursive descent for declarations, statements and patterns; precedence
#: climbing for expressions, with the table of spec §4.1. This is the mirror of
#: `src/lang/parser.c` and must agree with it node for node — `jaithon
#: --emit=ast --front=c` and `--front=jai` print the same JSON for every file
#: under `lib/` and `tests/`, and `make bootstrap` fails when they diverge.
#:
#: ```
#: let program = parse("let x = 1\n")
#: program?.kind == NodeKind.Program     # true
#: parse("let x =\n") == null            # true: a file with an error has no tree
#: ```
#:
#: Error recovery: exactly one diagnostic per synchronisation window. Panic mode
#: suppresses everything until `_synchronize` finds the next statement boundary,
#: so N independent syntax errors produce N diagnostics rather than a cascade.
#: A file with any error parses to null; the tree of a broken file is not worth
#: handing downstream.
from jaithon.ast import CompKind, DeclKind, Node, NodeKind, OpKind, Span, TypeExpr, TypeKind
from jaithon.ast import Visibility
from jaithon.compile.lexer import Code, Diagnostic, DiagnosticBag, Lexer, Token, TokenFlags
from jaithon.compile.lexer import TokenKind, kind_text

#: How deeply the recursive productions may nest before `E0107` stops them.
#:
#: The limit exists so that a file of ten thousand `(` characters reports a
#: syntax error instead of exhausting the call stack.
pub const MAX_RECURSION: int = 200

# Matches JAI_MAX_ARGS in src/common/common.h: the call instruction encodes the
# argument count in one byte.
const _MAX_ARGS = 255

# The two readings a prompt line has, and the verdict that it has neither yet.
const _REPL_AS_STATEMENT = 0
const _REPL_AS_EXPRESSION = 1
const _REPL_NEED_MORE = 2

#: The parser.
#:
#: Construct one over a `Lexer` and call `parse_program`. The parser executes
#: nothing, opens nothing, and keeps no state beyond its own fields; it reads
#: the finished token list and writes nodes.
pub class Parser {
    #: The token stream being read.
    pub let tokens: list[Token]
    #: Where diagnostics go; shared with the lexer that produced the tokens.
    pub var diagnostics: DiagnosticBag
    #: True once any syntax error has been reported.
    pub var had_error: bool

    let lexer: Lexer
    var current: int
    var panic_mode: bool
    var expr_depth: int
    var yield_count: int

    # Trial parsing, used only by `parse_repl_line`. A trial reads the line one
    # way, records whether that reading failed and on which token, and is then
    # undone. The token matters more than the failure: a diagnostic that fired
    # on `Eof` means the input stopped early and the next line can finish it,
    # while one that fired on a real token means the input is wrong and no
    # later line can undo it. Diagnostics are swallowed while a trial runs —
    # the reading that wins reports its own.
    var trial_active: bool
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
            let declaration = self._parse_declaration()
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
        let expr = self._parse_expression()
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
    #: Answers `(node, incomplete)`. `incomplete` says the input stopped early
    #: and a later line can finish it, so the prompt should keep reading; the
    #: node is null then. A null node with `incomplete` false is a line that
    #: was rejected, and the reason is in the bag.
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

    # Decided by parsing, because a count of open brackets cannot tell an input
    # that stopped early from one the parser rejected: it says "incomplete" for
    # both, and a line with a syntax error then holds the prompt open and
    # swallows every line typed after it.
    fn _choose_repl_reading(self) -> int {
        let (statement_ok, statement_ran_out, statement_tok) = self._repl_trial(false)
        if statement_ok { return _REPL_AS_STATEMENT }
        if statement_ran_out { return _REPL_NEED_MORE }
        if not self._check(TokenKind.LBrace) { return _REPL_AS_STATEMENT }

        let (expression_ok, expression_ran_out, expression_tok) = self._repl_trial(true)
        if expression_ok { return _REPL_AS_EXPRESSION }
        if expression_ran_out { return _REPL_NEED_MORE }

        # Neither reading works, so the line is wrong either way. Report the one
        # that got further, which is the one it was more nearly written as; a
        # tie goes to the block, so the prompt says what a file would say.
        if expression_tok > statement_tok { return _REPL_AS_EXPRESSION }
        return _REPL_AS_STATEMENT
    }

    # Read the line one way, learn how that ended, and undo all of it. Answers
    # `(ok, ran_out, error_tok)`, where `error_tok` doubles as how far the
    # reading got. Nodes a trial allocated are simply dropped.
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
        let ran_out = self.trial_failed and
            self._token_at(error_tok).kind == TokenKind.Eof
        self.trial_active = false

        self.current = mark_current
        self.expr_depth = mark_expr_depth
        self.yield_count = mark_yield_count
        self.had_error = mark_had_error
        self.panic_mode = mark_panic_mode
        return (ok, ran_out, error_tok)
    }

    # The reading a file gives the line, and the one nearly every line gets.
    fn _read_repl_statement(self) -> Node? {
        let node = self._parse_declaration()
        if self.panic_mode { self._synchronize() }
        self._skip_newlines()
        self._expect_repl_end()
        if self.had_error { return null }
        return node
    }

    # The reading only the prompt has. A statement opening with `{` is a block,
    # so in a file `{"a": 1}` is a block whose first statement is a string, and
    # it has to stay one; at the prompt a dict or a set literal would otherwise
    # be the one kind of value that cannot be typed.
    fn _read_repl_expression(self) -> Node? {
        let start_index = self.current
        let expr = self._parse_expression()
        if self.panic_mode { self._synchronize() }
        self._skip_newlines()
        self._expect_repl_end()
        if self.had_error { return null }
        return _node(NodeKind.ExprStmt, self._span_since(start_index), {"expr": expr})
    }

    # Nothing may follow a prompt line but newlines.
    fn _expect_repl_end(self) -> void {
        if self._check(TokenKind.Eof) or self.had_error { return }
        self._error_at_cur(
            Code.E0108_TRAILING_TOKENS,
            f"unexpected {self._describe(self._peek())} after the statement"
        )
    }

    # --- token access ----------------------------------------------------
    fn _token_at(self, index: int) -> Token {
        if index < 0 or index >= self.tokens.len() {
            return self.tokens[self.tokens.len() - 1]  # always the Eof token
        }
        return self.tokens[index]
    }

    fn _peek(self) -> Token {
        return self._token_at(self.current)
    }

    fn _kind(self) -> TokenKind {
        return self._peek().kind
    }

    fn _kind_at(self, offset: int) -> TokenKind {
        return self._token_at(self.current + offset).kind
    }

    fn _check(self, kind: TokenKind) -> bool {
        return self._kind() == kind
    }

    fn _advance(self) -> Token {
        let token = self._peek()
        if token.kind != TokenKind.Eof {
            self.current += 1
            self._skip_error_tokens()
        }
        return token
    }

    fn _match(self, kind: TokenKind) -> bool {
        if not self._check(kind) { return false }
        self._advance()
        return true
    }

    fn _skip_newlines(self) -> void {
        while self._check(TokenKind.Newline) { self._advance() }
    }

    # Lexical errors were already reported by the lexer; the parser steps over
    # the placeholder tokens without reporting them a second time.
    fn _skip_error_tokens(self) -> void {
        while self._token_at(self.current).kind == TokenKind.Error {
            self.had_error = true
            self.current += 1
        }
    }

    fn _cur_span(self) -> Span {
        return self._peek().span
    }

    # The span covering tokens [start_index, current), ignoring trailing
    # newlines so that a statement's span stops at its last real token.
    fn _span_since(self, start_index: int) -> Span {
        var end_index = self.current - 1
        while end_index > start_index and self._token_at(end_index).kind == TokenKind.Newline {
            end_index -= 1
        }
        if end_index < start_index { end_index = start_index }
        return _join(self._token_at(start_index).span, self._token_at(end_index).span)
    }

    fn _ident_text(self, offset: int, text: str) -> bool {
        let token = self._token_at(self.current + offset)
        return token.kind == TokenKind.Ident and token.text == text
    }

    # Label tokens carry their leading quote; the tree stores the bare name.
    fn _label_name(self, token: Token) -> str {
        if token.text.len() > 0 and token.text[0] == "'" { return token.text[1:token.text.len()] }
        return token.text
    }

    # --- diagnostics -----------------------------------------------------
    fn _describe(self, token: Token) -> str {
        if token.kind == TokenKind.Eof { return "end of file" }
        if token.kind == TokenKind.Newline { return "a newline" }
        if token.text.len() == 0 { return f"`{kind_text(token.kind)}`" }
        if token.text.len() > 32 { return f"`{token.text[0:32]}`" }
        return f"`{token.text}`"
    }

    # The single error entry point. Returns null when the diagnostic was
    # suppressed, so a caller that wants to attach a label must check.
    fn _error_at(
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

    # Remember the first diagnostic of a trial and swallow it.
    fn _trial_note_error(self) -> Diagnostic? {
        if not self.trial_failed {
            self.trial_failed = true
            self.trial_error_tok = self.current
        }
        return null
    }

    fn _error_at_cur(self, code: str, message: str, help: str? = null) -> Diagnostic? {
        return self._error_at(self._cur_span(), code, message, help: help)
    }

    # A construct we understood well enough to keep parsing: report it, but stay
    # out of panic mode so that the rest of the line still produces good
    # diagnostics. This is what the Jaithon 2 spellings use.
    fn _error_recovered(
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

    fn _expect(self, kind: TokenKind, what: str) -> bool {
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

    # Report E0107 with a label on the delimiter that was never closed.
    fn _unclosed(self, open_index: int, close_text: str, what: str) -> void {
        let diag = self._error_at(
            self._cur_span(),
            Code.E0107_UNCLOSED_DELIMITER,
            f"expected `{close_text}` to close this {what}, " +
                f"found {self._describe(self._peek())}"
        )
        if diag != null {
            diag.label(self._token_at(open_index).span, f"unclosed {what} opened here")
        }
    }

    fn _expect_ident_name(self, what: str) -> str? {
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

    # Placeholder returned instead of null so that no expression caller has to
    # check; the tree is discarded anyway once `had_error` is set.
    fn _error_expr(self, span: Span) -> Node {
        return _node(NodeKind.NullLit, span)
    }

    fn _enter_recursion(self) -> bool {
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

    # Skip to the next statement boundary at the current brace depth.
    fn _synchronize(self) -> void {
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

    fn _end_statement(self) -> void {
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

    # --- types -----------------------------------------------------------
    fn _parse_primary_type(self) -> TypeExpr {
        let start_index = self.current

        if self._check(TokenKind.KwFn) {
            self._advance()
            var params: list[TypeExpr] = []
            self._expect(TokenKind.LParen, "`(` after `fn` in a function type")
            self._skip_newlines()
            if not self._check(TokenKind.RParen) {
                loop {
                    self._skip_newlines()
                    params.push(self._parse_type())
                    self._skip_newlines()
                    if not self._match(TokenKind.Comma) { break }
                    if self._check(TokenKind.RParen) or self._check(TokenKind.Eof) { break }
                }
            }
            self._expect(TokenKind.RParen, "`)` to close the parameter list")
            if not self._match(TokenKind.Arrow) {
                self._error_at_cur(
                    Code.E0101_EXPECTED_TOKEN,
                    "expected `->` and a return type in a function type"
                )
                let missing = _type(TypeKind.Infer, self._cur_span())
                return _type(
                    TypeKind.Fn,
                    self._span_since(start_index),
                    args: params,
                    inner: missing
                )
            }
            let result = self._parse_type()
            return _type(TypeKind.Fn, self._span_since(start_index), args: params, inner: result)
        }

        if self._check(TokenKind.LParen) {
            self._advance()
            var members: list[TypeExpr] = []
            var trailing_comma = false
            self._skip_newlines()
            if not self._check(TokenKind.RParen) {
                loop {
                    self._skip_newlines()
                    members.push(self._parse_type())
                    self._skip_newlines()
                    if not self._match(TokenKind.Comma) { break }
                    self._skip_newlines()
                    if self._check(TokenKind.RParen) {
                        trailing_comma = true
                        break
                    }
                    if self._check(TokenKind.Eof) { break }
                }
            }
            self._expect(TokenKind.RParen, "`)` to close the tuple type")
            if members.len() == 1 and not trailing_comma { return members[0] }
            return _type(TypeKind.Tuple, self._span_since(start_index), args: members)
        }

        if self._match(TokenKind.Underscore) {
            return _type(TypeKind.Infer, self._span_since(start_index))
        }
        if self._match(TokenKind.KwNull) {
            return _type(TypeKind.Name, self._span_since(start_index), name: "null")
        }

        if not self._check(TokenKind.Ident) {
            self._error_at(
                self._cur_span(),
                Code.E0105_EXPECTED_TYPE,
                f"expected a type, found {self._describe(self._peek())}"
            )
            return _type(TypeKind.Infer, self._cur_span())
        }

        # A dotted name is a module-qualified type such as `std.math.Vec`.
        var name = self._advance().text
        while self._check(TokenKind.Dot) and self._kind_at(1) == TokenKind.Ident {
            self._advance()
            name = name + "." + self._advance().text
        }

        if not self._check(TokenKind.LBracket) {
            return _type(TypeKind.Name, self._span_since(start_index), name: name)
        }

        let open_index = self.current
        self._advance()
        var args: list[TypeExpr] = []
        self._skip_newlines()
        if not self._check(TokenKind.RBracket) {
            loop {
                self._skip_newlines()
                args.push(self._parse_type())
                self._skip_newlines()
                if not self._match(TokenKind.Comma) { break }
                self._skip_newlines()
                if self._check(TokenKind.RBracket) or self._check(TokenKind.Eof) { break }
            }
        }
        if self._check(TokenKind.RBracket) {
            self._advance()
        } else {
            self._unclosed(open_index, "]", "type argument list")
        }
        return _type(TypeKind.Generic, self._span_since(start_index), name: name, args: args)
    }

    fn _parse_postfix_type(self) -> TypeExpr {
        let start_index = self.current
        var result = self._parse_primary_type()
        while self._match(TokenKind.Question) {
            result = _type(TypeKind.Optional, self._span_since(start_index), inner: result)
        }
        return result
    }

    fn _parse_type(self) -> TypeExpr {
        if not self._enter_recursion() {
            self.expr_depth -= 1
            return _type(TypeKind.Infer, self._cur_span())
        }
        defer { self.expr_depth -= 1 }

        let start_index = self.current
        let first = self._parse_postfix_type()
        if not self._check(TokenKind.Pipe) { return first }

        var parts: list[TypeExpr] = [first]
        while self._match(TokenKind.Pipe) {
            parts.push(self._parse_postfix_type())
            if self._check(TokenKind.Eof) { break }
        }
        return _type(TypeKind.Union, self._span_since(start_index), args: parts)
    }

    # `catch e: A | B` and `class C: A, B` both flatten into a type list.
    fn _parse_type_list(self) -> list[TypeExpr] {
        var types: list[TypeExpr] = []
        loop {
            let parsed = self._parse_type()
            if parsed.kind == TypeKind.Union {
                for member in parsed.args { types.push(member) }
            } else {
                types.push(parsed)
            }
            if not self._match(TokenKind.Comma) { break }
            if self._check(TokenKind.Eof) or self.panic_mode { break }
        }
        return types
    }

    # --- arguments and f-strings -----------------------------------------

    # Parse an argument list up to and including the closing `)`.
    fn _parse_args(self, open_index: int) -> list[dict[str, any]] {
        var args: list[dict[str, any]] = []
        self._skip_newlines()
        if not self._check(TokenKind.RParen) {
            loop {
                self._skip_newlines()
                let arg_start = self.current
                var name: str? = null
                let is_spread = self._match(TokenKind.Ellipsis)
                # `name: value` is a keyword argument; nothing else puts a colon
                # directly after an identifier here.
                if not is_spread and
                    self._check(TokenKind.Ident) and
                    self._kind_at(1) == TokenKind.Colon {
                    name = self._advance().text
                    self._advance()
                }
                let value = self._parse_expression()
                let span = self._span_since(arg_start)
                args.push(_arg(name, value, is_spread, span))

                if args.len() > _MAX_ARGS {
                    self._error_at(
                        span,
                        Code.E0100_UNEXPECTED_TOKEN,
                        f"more than {_MAX_ARGS} arguments in one call"
                    )
                    break
                }
                self._skip_newlines()
                if not self._match(TokenKind.Comma) { break }
                self._skip_newlines()
                if self._check(TokenKind.RParen) or self._check(TokenKind.Eof) { break }
            }
        }
        if self._check(TokenKind.RParen) {
            self._advance()
        } else {
            self._unclosed(open_index, ")", "argument list")
        }
        return args
    }

    # A format spec (`{x:.2f}`) becomes an explicit `__format__` call, so that
    # code generation needs no special case for it.
    fn _apply_format_spec(self, value: Node) -> Node {
        if not self._match(TokenKind.Colon) { return value }
        let spec_span = self._cur_span()
        var spec_end = spec_span.start
        while not self._check(TokenKind.FStringMid) and
            not self._check(TokenKind.FStringEnd) and
            not self._check(TokenKind.Eof) {
            spec_end = self._advance().span.end
        }
        if spec_end <= spec_span.start { return value }

        let span = value.span
        let text = self.lexer.text_of(Span(spec_span.file, spec_span.start, spec_end))
        let formatter = _member(NodeKind.Member, span, value, "__format__")
        return _node(
            NodeKind.Call,
            span,
            {"callee": formatter, "args": [_arg(null, _str_lit(span, text), false, span)]}
        )
    }

    fn _parse_fstring(self) -> Node {
        let start_index = self.current
        var parts: list[Node] = []

        # An f-string with no holes arrives as a lone `FStringEnd` chunk.
        let start_token = self._advance()
        if start_token.string_value().len() > 0 {
            parts.push(_str_lit(start_token.span, start_token.string_value()))
        }
        let complete = start_token.kind == TokenKind.FStringEnd

        while not complete {
            if self._check(TokenKind.Eof) {
                self._error_at_cur(
                    Code.E0107_UNCLOSED_DELIMITER,
                    "unterminated interpolated string"
                )
                break
            }
            if self._check(TokenKind.Colon) {
                # An empty `{:spec}` is a hole with no expression.
                let empty = self._error_expr(self._cur_span())
                self._error_at_cur(
                    Code.E0102_EXPECTED_EXPRESSION,
                    "expected an expression before the format spec"
                )
                parts.push(empty)
            } elif not self._check(TokenKind.FStringMid) and not self._check(TokenKind.FStringEnd) {
                parts.push(self._apply_format_spec(self._parse_expression()))
            }

            if self._check(TokenKind.FStringMid) {
                let mid = self._advance()
                if mid.string_value().len() > 0 {
                    parts.push(_str_lit(mid.span, mid.string_value()))
                }
                continue
            }
            if self._check(TokenKind.FStringEnd) {
                let end = self._advance()
                if end.string_value().len() > 0 {
                    parts.push(_str_lit(end.span, end.string_value()))
                }
                break
            }
            if not self.panic_mode {
                self._error_at_cur(
                    Code.E0101_EXPECTED_TOKEN,
                    f"expected `}}` to close the interpolation, " +
                        f"found {self._describe(self._peek())}"
                )
            }
            break
        }

        let span = self._span_since(start_index)
        if parts.len() == 0 { parts.push(_str_lit(span, "")) }
        return _node(NodeKind.FString, span, {"parts": parts})
    }

    # --- comprehensions and collection literals --------------------------
    fn _parse_comp_clauses(self) -> list[dict[str, any]] {
        var clauses: list[dict[str, any]] = []
        while self._check(TokenKind.KwFor) {
            let clause_start = self.current
            self._advance()
            let pattern = self._parse_pattern()
            self._expect(TokenKind.KwIn, "`in` after the comprehension pattern")
            let iterable = self._parse_expression()

            var conditions: list[Node] = []
            while self._match(TokenKind.KwIf) {
                conditions.push(self._parse_expression())
                if self._check(TokenKind.Eof) { break }
            }
            clauses.push(
                _comp_clause(pattern, iterable, conditions, self._span_since(clause_start))
            )
            if self._check(TokenKind.Eof) or self.panic_mode { break }
        }
        return clauses
    }

    # Every comprehension is opened by the bracket at `start_index`, so that one
    # index is both where the node starts and what the unclosed-delimiter
    # diagnostic points at.
    fn _finish_comprehension(
        self,
        start_index: int,
        comp_kind: CompKind,
        key_expr: Node?,
        element: Node,
        closer: TokenKind,
        close_text: str,
        what: str
    ) -> Node {
        let clauses = self._parse_comp_clauses()
        self._skip_newlines()
        self._close_sequence(closer, close_text, what, start_index)

        return _node(
            NodeKind.Comprehension,
            self._span_since(start_index),
            {
                "compKind": comp_kind,
                "element": element,
                "keyExpr": key_expr,
                "clauses": clauses,
            }
        )
    }

    # Elements of a `[...]`, `{...}` or `(...)` literal after the first one.
    fn _parse_rest_of_sequence(self, first: Node, closer: TokenKind) -> list[Node] {
        var items: list[Node] = [first]
        self._skip_newlines()
        while self._match(TokenKind.Comma) {
            self._skip_newlines()
            if self._check(closer) or self._check(TokenKind.Eof) { break }
            items.push(self._parse_expression())
            self._skip_newlines()
            if self.panic_mode { break }
        }
        return items
    }

    fn _close_sequence(
        self,
        closer: TokenKind,
        close_text: str,
        what: str,
        open_index: int
    ) -> void {
        if self._check(closer) {
            self._advance()
        } else {
            self._unclosed(open_index, close_text, what)
        }
    }

    fn _parse_list_literal(self) -> Node {
        let start_index = self.current
        self._advance()  # `[`
        self._skip_newlines()

        if self._match(TokenKind.RBracket) {
            return _sequence(NodeKind.ListLit, self._span_since(start_index), [])
        }

        let first = self._parse_expression()
        if self._check(TokenKind.KwFor) {
            return self._finish_comprehension(
                start_index,
                CompKind.List,
                null,
                first,
                TokenKind.RBracket,
                "]",
                "list comprehension"
            )
        }
        let items = self._parse_rest_of_sequence(first, TokenKind.RBracket)
        self._close_sequence(TokenKind.RBracket, "]", "list literal", start_index)
        return _sequence(NodeKind.ListLit, self._span_since(start_index), items)
    }

    # `{}` is the empty dict, `{a, b}` a set, and `{k: v}` a dict.
    fn _parse_brace_literal(self) -> Node {
        let start_index = self.current
        self._advance()  # `{`
        self._skip_newlines()

        if self._match(TokenKind.RBrace) {
            return _dict_lit(self._span_since(start_index), [], [])
        }

        let first = self._parse_expression()

        if self._match(TokenKind.Colon) {
            self._skip_newlines()
            let first_value = self._parse_expression()
            if self._check(TokenKind.KwFor) {
                return self._finish_comprehension(
                    start_index,
                    CompKind.Dict,
                    first,
                    first_value,
                    TokenKind.RBrace,
                    "}",
                    "dict comprehension"
                )
            }
            var keys: list[Node] = [first]
            var values: list[Node] = [first_value]
            self._skip_newlines()
            while self._match(TokenKind.Comma) {
                self._skip_newlines()
                if self._check(TokenKind.RBrace) or self._check(TokenKind.Eof) { break }
                keys.push(self._parse_expression())
                self._expect(TokenKind.Colon, "`:` between a dict key and its value")
                self._skip_newlines()
                values.push(self._parse_expression())
                self._skip_newlines()
                if self.panic_mode { break }
            }
            self._close_sequence(TokenKind.RBrace, "}", "dict literal", start_index)
            return _dict_lit(self._span_since(start_index), keys, values)
        }

        if self._check(TokenKind.KwFor) {
            return self._finish_comprehension(
                start_index,
                CompKind.Set,
                null,
                first,
                TokenKind.RBrace,
                "}",
                "set comprehension"
            )
        }
        let items = self._parse_rest_of_sequence(first, TokenKind.RBrace)
        self._close_sequence(TokenKind.RBrace, "}", "set literal", start_index)
        return _sequence(NodeKind.SetLit, self._span_since(start_index), items)
    }

    fn _parse_paren_expr(self) -> Node {
        let start_index = self.current
        self._advance()  # `(`
        self._skip_newlines()

        if self._match(TokenKind.RParen) {
            return _sequence(NodeKind.TupleLit, self._span_since(start_index), [])
        }

        let first = self._parse_expression()
        self._skip_newlines()

        if self._check(TokenKind.KwFor) {
            return self._finish_comprehension(
                start_index,
                CompKind.Generator,
                null,
                first,
                TokenKind.RParen,
                ")",
                "generator expression"
            )
        }

        if self._check(TokenKind.Comma) {
            let items = self._parse_rest_of_sequence(first, TokenKind.RParen)
            self._close_sequence(TokenKind.RParen, ")", "tuple literal", start_index)
            return _sequence(NodeKind.TupleLit, self._span_since(start_index), items)
        }

        self._close_sequence(TokenKind.RParen, ")", "parenthesised expression", start_index)
        return first
    }

    # --- lambdas ---------------------------------------------------------
    fn _parse_lambda(self) -> Node {
        let start_index = self.current
        self._advance()  # `|`

        var params: list[dict[str, any]] = []
        if not self._check(TokenKind.Pipe) {
            loop {
                let name_token = self._peek()
                let name = self._expect_ident_name("a lambda parameter name")
                if name == null { break }
                var declared: TypeExpr? = null
                if self._match(TokenKind.Colon) { declared = self._parse_type() }
                self._check_duplicate_param(params, name, name_token.span)
                params.push(_param(name, declared, null, name_token.span, false, false))
                if not self._match(TokenKind.Comma) { break }
                if self._check(TokenKind.Eof) { break }
            }
        }
        self._expect(TokenKind.Pipe, "`|` to close the lambda parameter list")

        let saved_yield = self.yield_count
        self.yield_count = 0
        let body = self._parse_expression()
        let is_generator = self.yield_count > 0
        self.yield_count = saved_yield

        return _node(
            NodeKind.Lambda,
            self._span_since(start_index),
            {
                "name": null,
                "generics": [],
                "params": params,
                "returnType": null,
                "body": body,
                "visibility": Visibility.Private,
                "isStatic": false,
                "isExprBody": true,
                "isGenerator": is_generator,
                "isAsync": false,
            }
        )
    }

    # --- primary ---------------------------------------------------------
    fn _parse_primary(self) -> Node {
        let start_index = self.current
        let token = self._peek()
        let kind = token.kind

        if kind == TokenKind.Int {
            self._advance()
            return _node(NodeKind.IntLit, token.span, {"intLit": token.value ?? 0})
        }
        if kind == TokenKind.Float {
            self._advance()
            return _node(NodeKind.FloatLit, token.span, {"floatLit": token.value ?? 0.0})
        }
        if kind == TokenKind.Str {
            self._advance()
            return _str_lit(token.span, token.string_value())
        }
        if kind == TokenKind.FStringStart or kind == TokenKind.FStringEnd {
            return self._parse_fstring()
        }
        # `throw e` is an expression of type `never` (§4.3), which is what lets a
        # match arm or the right-hand side of `??` raise. The statement form is
        # matched earlier, so this only fires where an expression is expected.
        if kind == TokenKind.KwThrow {
            self._advance()
            let operand = self._parse_expression()
            return _wrap(NodeKind.ThrowExpr, self._span_since(start_index), operand)
        }
        if kind == TokenKind.KwTrue or kind == TokenKind.KwFalse {
            self._advance()
            return _node(NodeKind.BoolLit, token.span, {"boolLit": kind == TokenKind.KwTrue})
        }
        if kind == TokenKind.KwNull {
            self._advance()
            return _node(NodeKind.NullLit, token.span)
        }
        if kind == TokenKind.KwSelf {
            self._advance()
            return _node(NodeKind.Self, token.span)
        }
        if kind == TokenKind.KwSuper {
            self._advance()
            return _node(NodeKind.Super, token.span)
        }
        if kind == TokenKind.Underscore {
            self._advance()
            return _ident(token.span, "_")
        }
        if kind == TokenKind.LParen { return self._parse_paren_expr() }
        if kind == TokenKind.LBracket { return self._parse_list_literal() }
        if kind == TokenKind.LBrace { return self._parse_brace_literal() }
        if kind == TokenKind.Pipe { return self._parse_lambda() }
        if kind == TokenKind.KwIf { return self._parse_if(true) }
        if kind == TokenKind.KwMatch { return self._parse_match(true) }
        if kind == TokenKind.KwFn {
            let anonymous = self._parse_fn_rest(
                Visibility.Private,
                false,
                start_index,
                false,
                false
            )
            return anonymous ?? self._error_expr(token.span)
        }
        if kind == TokenKind.KwYield {
            self._advance()
            self.yield_count += 1
            let operand = self._starts_expression() ? self._parse_expression() : null
            return _node(NodeKind.Yield, self._span_since(start_index), {"operand": operand})
        }
        if kind == TokenKind.Ident { return self._parse_ident_primary(token, start_index) }

        let diag = self._error_at(
            token.span,
            Code.E0102_EXPECTED_EXPRESSION,
            f"expected an expression, found {self._describe(token)}"
        )
        if diag != null {
            if kind == TokenKind.KwElse or kind == TokenKind.KwElif {
                diag.with_help(f"put `{token.text}` on the same line as the closing `}}`")
            } elif kind == TokenKind.Eq {
                diag.with_help("`==` compares, `=` assigns")
            }
        }
        return self._error_expr(token.span)
    }

    fn _parse_ident_primary(self, token: Token, start_index: int) -> Node {
        # `new Foo()` — the keyword was removed in Jaithon 3.
        if token.text == "new" and self._kind_at(1) == TokenKind.Ident {
            let class_name = self._token_at(self.current + 1).text
            self._error_recovered(
                token.span,
                Code.E0100_UNEXPECTED_TOKEN,
                "the `new` keyword was removed in Jaithon 3",
                help: f"a class is callable: write `{class_name}(...)`"
            )
            self._advance()
            return self._parse_primary()
        }
        # `await expr` is contextual: `std.async` defines it, so it is only an
        # operator when an operand plainly follows.
        if token.text == "await" and _starts_await_operand(self._kind_at(1)) {
            self._advance()
            let operand = self._parse_unary()
            return _wrap(NodeKind.Await, self._span_since(start_index), operand)
        }
        self._advance()
        return _ident(token.span, token.text)
    }

    # --- postfix ---------------------------------------------------------
    fn _parse_subscript(self, object: Node, open_index: int) -> Node {
        self._advance()  # `[`
        var start: Node? = null
        if not self._check(TokenKind.Colon) {
            if self._check(TokenKind.RBracket) {
                self._error_at_cur(Code.E0102_EXPECTED_EXPRESSION, "expected an index expression")
                self._advance()
                return self._error_expr(_join(object.span, self._span_since(open_index)))
            }
            start = self._parse_expression()
        }

        if self._match(TokenKind.Colon) {
            var stop: Node? = null
            var step: Node? = null
            if not self._check(TokenKind.Colon) and not self._check(TokenKind.RBracket) {
                stop = self._parse_expression()
            }
            if self._match(TokenKind.Colon) {
                if not self._check(TokenKind.RBracket) { step = self._parse_expression() }
            }
            self._close_sequence(TokenKind.RBracket, "]", "slice", open_index)
            return _node(
                NodeKind.Slice,
                _join(object.span, self._span_since(open_index)),
                {"object": object, "start": start, "stop": stop, "step": step}
            )
        }

        # `x[a, b]` is never a valid index — an index takes exactly one
        # expression — so a comma here means generic type arguments:
        # `Pair[B, A](x, y)`. The list becomes a tuple and stays an `Index`;
        # only the symbol table knows whether `Pair` names a class, so the
        # resolver decides which it is. Keeping that decision out of the parser
        # is what makes `Box[int]` unambiguous without lookahead hacks.
        if self._check(TokenKind.Comma) {
            var args: list[Node] = [start ?? self._error_expr(self._cur_span())]
            while self._match(TokenKind.Comma) {
                self._skip_newlines()
                if self._check(TokenKind.RBracket) { break }  # trailing comma
                args.push(self._parse_expression())
                if self.panic_mode { break }
            }
            self._close_sequence(TokenKind.RBracket, "]", "generic argument list", open_index)
            let generic_span = _join(object.span, self._span_since(open_index))
            let tuple = _node(NodeKind.TupleLit, generic_span, {"items": args})
            return _node(NodeKind.Index, generic_span, {"object": object, "index": tuple})
        }

        self._close_sequence(TokenKind.RBracket, "]", "index expression", open_index)
        let span = _join(object.span, self._span_since(open_index))
        return _node(
            NodeKind.Index,
            span,
            {"object": object, "index": start ?? self._error_expr(span)}
        )
    }

    fn _parse_postfix(self) -> Node {
        var expr = self._parse_primary()

        loop {
            if self._check(TokenKind.Dot) or self._check(TokenKind.QuestionDot) {
                let optional = self._check(TokenKind.QuestionDot)
                self._advance()
                let name_token = self._peek()
                let name = self._expect_ident_name("a member name after `.`")
                if name == null { break }
                let kind = optional ? NodeKind.OptMember : NodeKind.Member
                expr = _member(kind, _join(expr.span, name_token.span), expr, name)
            } elif self._check(TokenKind.LParen) {
                let open_index = self.current
                self._advance()
                let args = self._parse_args(open_index)
                expr = _node(
                    NodeKind.Call,
                    _join(expr.span, self._span_since(open_index)),
                    {"callee": expr, "args": args}
                )
            } elif self._check(TokenKind.LBracket) {
                expr = self._parse_subscript(expr, self.current)
            } else {
                break
            }
            if self.panic_mode { break }
        }
        return expr
    }

    # --- the precedence ladder (§4.1) ------------------------------------
    fn _parse_power(self) -> Node {
        let left = self._parse_postfix()
        if not self._match(TokenKind.StarStar) { return left }
        # Right-associative, and looser than a unary on its right so that
        # `2 ** -1` parses.
        return _binary(NodeKind.Binary, OpKind.Pow, left, self._parse_unary())
    }

    # `-9223372036854775808` is the one way to write the smallest `int`, so the
    # `-` in front of a NEG_MAGNITUDE literal belongs to the literal: the lexer
    # already stored the value negated, and negating it again is not
    # representable. Matching the token pair — rather than any IntLit that
    # happens to equal the smallest `int` — spends the sign exactly once, so
    # `-(-9223372036854775808)` still raises. Call with the cursor on the `-`.
    fn _parse_neg_magnitude(self) -> Node? {
        let literal = self._token_at(self.current + 1)
        if literal.kind != TokenKind.Int or not literal.has_flag(TokenFlags.NEG_MAGNITUDE) {
            return null
        }
        let start_index = self.current
        self._advance()  # the `-`
        self._advance()  # the literal
        return _node(NodeKind.IntLit, self._span_since(start_index), {"intLit": literal.value ?? 0})
    }

    fn _parse_unary(self) -> Node {
        if not self._enter_recursion() {
            self.expr_depth -= 1
            return self._error_expr(self._cur_span())
        }
        defer { self.expr_depth -= 1 }

        let kind = self._kind()
        if kind == TokenKind.Minus {
            let folded = self._parse_neg_magnitude()
            if folded != null { return folded ?? self._error_expr(self._cur_span()) }
        }
        if kind != TokenKind.Minus and kind != TokenKind.Plus and kind != TokenKind.Tilde {
            return self._parse_power()
        }
        let start_index = self.current
        self._advance()
        let operand = self._parse_unary()
        var op = OpKind.Bnot
        if kind == TokenKind.Minus {
            op = OpKind.Neg
        } elif kind == TokenKind.Plus {
            op = OpKind.Pos
        }
        return _unary(self._span_since(start_index), op, operand)
    }

    fn _parse_multiplicative(self) -> Node {
        var left = self._parse_unary()
        loop {
            let op = _multiplicative_op(self._kind())
            if op == null { return left }
            self._advance()
            left = _binary(NodeKind.Binary, op ?? OpKind.Mul, left, self._parse_unary())
            if self.panic_mode { return left }
        }
    }

    fn _parse_additive(self) -> Node {
        var left = self._parse_multiplicative()
        loop {
            let op = _additive_op(self._kind())
            if op == null { return left }
            self._advance()
            left = _binary(NodeKind.Binary, op ?? OpKind.Add, left, self._parse_multiplicative())
            if self.panic_mode { return left }
        }
    }

    fn _parse_shift(self) -> Node {
        var left = self._parse_additive()
        loop {
            let kind = self._kind()
            if kind != TokenKind.Shl and kind != TokenKind.Shr { return left }
            self._advance()
            let op = kind == TokenKind.Shl ? OpKind.Shl : OpKind.Shr
            left = _binary(NodeKind.Binary, op, left, self._parse_additive())
            if self.panic_mode { return left }
        }
    }

    fn _parse_bit_and(self) -> Node {
        var left = self._parse_shift()
        while self._match(TokenKind.Amp) {
            left = _binary(NodeKind.Binary, OpKind.Band, left, self._parse_shift())
            if self.panic_mode { break }
        }
        return left
    }

    fn _parse_bit_xor(self) -> Node {
        var left = self._parse_bit_and()
        while self._check(TokenKind.Caret) {
            let op_span = self._cur_span()
            self._advance()
            let right = self._parse_bit_and()
            # Two numeric literals around `^` is almost always a Jaithon 2 power.
            #
            # Suppressed during a trial for the reason a trial's errors are: the
            # reading that wins reports its own. The REPL reads every line as a
            # trial and then again for real, so a warning that reached the bag
            # from a trial is rendered twice — three times for a line opening
            # with `{`, which is tried both ways.
            if not self.trial_active and
                _is_numeric_literal(left) and _is_numeric_literal(right) {
                self.diagnostics.warn(
                    Code.W0105_DEPRECATED,
                    op_span,
                    "`^` is bitwise xor in Jaithon 3, not exponentiation",
                    note: "the power operator is `**`"
                )
            }
            left = _binary(NodeKind.Binary, OpKind.Bxor, left, right)
            if self.panic_mode { break }
        }
        return left
    }

    fn _parse_bit_or(self) -> Node {
        var left = self._parse_bit_xor()
        while self._match(TokenKind.Pipe) {
            left = _binary(NodeKind.Binary, OpKind.Bor, left, self._parse_bit_xor())
            if self.panic_mode { break }
        }
        return left
    }

    fn _parse_range(self) -> Node {
        let left = self._parse_bit_or()
        if not self._check(TokenKind.DotDot) and not self._check(TokenKind.DotDotEq) {
            return left
        }
        let inclusive = self._check(TokenKind.DotDotEq)
        self._advance()
        let stop = self._starts_expression() ? self._parse_bit_or() : null
        var span = left.span
        if stop != null { span = _join(left.span, stop.span) }
        return _node(NodeKind.Range, span, {"start": left, "stop": stop, "inclusive": inclusive})
    }

    # The level-6 operator at the cursor, as `(op, token count, is legacy)`.
    fn _peek_compare_op(self) -> tuple[OpKind, int, bool]? {
        let kind = self._kind()
        let simple = _compare_op(kind)
        if simple != null { return (simple ?? OpKind.Eq, 1, false) }
        if kind == TokenKind.KwIs {
            if self._kind_at(1) == TokenKind.KwNot { return (OpKind.IsNot, 2, false) }
            return (OpKind.Is, 1, false)
        }
        if kind == TokenKind.KwNot {
            if self._kind_at(1) == TokenKind.KwIn { return (OpKind.NotIn, 2, false) }
            return null
        }
        if kind == TokenKind.Ident {
            let text = self._peek().text
            if text == "eq" { return (OpKind.Eq, 1, true) }
            if text == "ne" or text == "neq" { return (OpKind.Ne, 1, true) }
        }
        return null
    }

    # `a < b <= c` is one chain node, with `b` evaluated once (§4.2).
    fn _parse_comparison(self) -> Node {
        let first = self._parse_range()
        if self._peek_compare_op() == null { return first }

        var operands: list[Node] = [first]
        var ops: list[OpKind] = []
        loop {
            let found = self._peek_compare_op()
            if found == null { break }
            let (op, op_tokens, legacy) = found
            if legacy {
                let spelling = op == OpKind.Eq ? "==" : "!="
                self._error_recovered(
                    self._cur_span(),
                    Code.E0100_UNEXPECTED_TOKEN,
                    f"`{self._peek().text}` is not an operator in Jaithon 3",
                    help: f"write `{spelling}`"
                )
            }
            for _step in 0..op_tokens { self._advance() }
            ops.push(op)
            operands.push(self._parse_range())
            if self.panic_mode or self._check(TokenKind.Eof) { break }
        }

        if ops.len() == 1 { return _binary(NodeKind.Binary, ops[0], operands[0], operands[1]) }
        let last = operands[operands.len() - 1]
        return _node(
            NodeKind.CompareChain,
            _join(first.span, last.span),
            {"operands": operands, "ops": ops}
        )
    }

    fn _parse_not(self) -> Node {
        if not self._check(TokenKind.KwNot) { return self._parse_comparison() }
        if not self._enter_recursion() {
            self.expr_depth -= 1
            return self._error_expr(self._cur_span())
        }
        defer { self.expr_depth -= 1 }

        let start_index = self.current
        self._advance()
        let operand = self._parse_not()
        return _unary(self._span_since(start_index), OpKind.Not, operand)
    }

    fn _parse_and(self) -> Node {
        var left = self._parse_not()
        while self._match(TokenKind.KwAnd) {
            left = _binary(NodeKind.Logical, OpKind.And, left, self._parse_not())
            if self.panic_mode { break }
        }
        return left
    }

    fn _parse_or(self) -> Node {
        var left = self._parse_and()
        while self._match(TokenKind.KwOr) {
            left = _binary(NodeKind.Logical, OpKind.Or, left, self._parse_and())
            if self.panic_mode { break }
        }
        return left
    }

    fn _parse_coalesce(self) -> Node {
        var left = self._parse_or()
        while self._match(TokenKind.QuestionQuestion) {
            let right = self._parse_or()
            left = _node(
                NodeKind.Coalesce,
                _join(left.span, right.span),
                {"left": left, "right": right}
            )
            if self.panic_mode { break }
        }
        return left
    }

    fn _parse_expression(self) -> Node {
        if not self._enter_recursion() {
            self.expr_depth -= 1
            return self._error_expr(self._cur_span())
        }
        defer { self.expr_depth -= 1 }

        let cond = self._parse_coalesce()
        if not self._match(TokenKind.Question) { return cond }
        let then_expr = self._parse_expression()
        self._expect(TokenKind.Colon, "`:` in a conditional expression")
        let else_expr = self._parse_expression()
        return _node(
            NodeKind.Ternary,
            _join(cond.span, else_expr.span),
            {"cond": cond, "thenExpr": then_expr, "elseExpr": else_expr}
        )
    }

    fn _starts_expression(self) -> bool {
        return _starts_expression_kind(self._kind())
    }

    # --- patterns --------------------------------------------------------

    # Elements of a `(...)` or `[...]` pattern, including one `...rest`.
    # Returns the elements, the index of the rest element or -1, and whether a
    # comma was seen, which is what tells `(x)` from `(x,)`.
    fn _parse_pattern_list(
        self,
        closer: TokenKind,
        close_text: str,
        what: str,
        open_index: int
    ) -> tuple[list[Node], int, bool] {
        var elems: list[Node] = []
        var rest_index = -1
        var saw_comma = false

        self._skip_newlines()
        if not self._check(closer) {
            loop {
                self._skip_newlines()
                if self._check(TokenKind.Ellipsis) {
                    let rest_span = self._cur_span()
                    self._advance()
                    var rest = _node(NodeKind.PatWildcard, rest_span)
                    if self._check(TokenKind.Ident) {
                        let name_token = self._advance()
                        rest = _pat_bind(_join(rest_span, name_token.span), name_token.text)
                    }
                    if rest_index >= 0 {
                        self._error_at(
                            rest.span,
                            Code.E0110_INVALID_PATTERN,
                            "a pattern may bind only one `...` rest element"
                        )
                    } else {
                        rest_index = elems.len()
                    }
                    elems.push(rest)
                } else {
                    elems.push(self._parse_pattern())
                }
                self._skip_newlines()
                if not self._match(TokenKind.Comma) { break }
                saw_comma = true
                self._skip_newlines()
                if self._check(closer) or self._check(TokenKind.Eof) { break }
                if self.panic_mode { break }
            }
        }
        self._close_sequence(closer, close_text, what, open_index)
        return (elems, rest_index, saw_comma)
    }

    fn _parse_pattern_literal_value(self) -> Node {
        let start_index = self.current
        if self._check(TokenKind.Minus) {
            let folded = self._parse_neg_magnitude()  # `-9223372036854775808`
            if folded != null { return folded ?? self._error_expr(self._cur_span()) }
        }
        if not self._match(TokenKind.Minus) { return self._parse_primary() }
        let inner = self._parse_primary()
        return _unary(self._span_since(start_index), OpKind.Neg, inner)
    }

    fn _parse_pattern_primary(self) -> Node {
        let start_index = self.current

        if self._match(TokenKind.Underscore) {
            return _node(NodeKind.PatWildcard, self._span_since(start_index))
        }

        if self._check(TokenKind.LParen) {
            self._advance()
            let (elems, rest_index, saw_comma) = self._parse_pattern_list(
                TokenKind.RParen,
                ")",
                "tuple pattern",
                start_index
            )
            if elems.len() == 1 and rest_index < 0 and not saw_comma { return elems[0] }
            return _pat_seq(NodeKind.PatTuple, self._span_since(start_index), elems, rest_index)
        }

        if self._check(TokenKind.LBracket) {
            self._advance()
            let (elems, rest_index, _saw_comma) = self._parse_pattern_list(
                TokenKind.RBracket,
                "]",
                "list pattern",
                start_index
            )
            return _pat_seq(NodeKind.PatList, self._span_since(start_index), elems, rest_index)
        }

        if _starts_pattern_literal(self._kind()) {
            let low = self._parse_pattern_literal_value()
            if not self._check(TokenKind.DotDot) and not self._check(TokenKind.DotDotEq) {
                return _node(NodeKind.PatLiteral, self._span_since(start_index), {"value": low})
            }
            let inclusive = self._check(TokenKind.DotDotEq)
            self._advance()
            var high: Node? = null
            if _starts_pattern_literal(self._kind()) { high = self._parse_pattern_literal_value() }
            return _node(
                NodeKind.PatRange,
                self._span_since(start_index),
                {"lo": low, "hi": high, "inclusive": inclusive}
            )
        }

        if self._check(TokenKind.Ident) {
            let name = self._advance().text

            # `Shape.Circle(r)` — an enum variant, with an optional payload.
            if self._match(TokenKind.Dot) {
                let variant = self._expect_ident_name("an enum variant name")
                if variant == null {
                    return _node(NodeKind.PatWildcard, self._span_since(start_index))
                }
                var subs: list[Node] = []
                if self._check(TokenKind.LParen) {
                    self._advance()
                    let (elems, _rest, _comma) = self._parse_pattern_list(
                        TokenKind.RParen,
                        ")",
                        "variant pattern",
                        start_index
                    )
                    subs = elems
                }
                return _pat_class(
                    NodeKind.PatEnum,
                    self._span_since(start_index),
                    name,
                    variant,
                    _nulls(subs.len()),
                    subs
                )
            }

            # `Point{x, y}` or `Point{x: 1, y}` — a class pattern.
            if self._check(TokenKind.LBrace) {
                return self._parse_class_pattern(name, start_index)
            }

            # `Some(x)` — a positional class pattern.
            if self._check(TokenKind.LParen) {
                self._advance()
                let (elems, _rest, _comma) = self._parse_pattern_list(
                    TokenKind.RParen,
                    ")",
                    "class pattern",
                    start_index
                )
                return _pat_class(
                    NodeKind.PatClass,
                    self._span_since(start_index),
                    name,
                    null,
                    _nulls(elems.len()),
                    elems
                )
            }

            return _pat_bind(self._span_since(start_index), name)
        }

        self._error_at(
            self._cur_span(),
            Code.E0110_INVALID_PATTERN,
            f"expected a pattern, found {self._describe(self._peek())}"
        )
        return _node(NodeKind.PatWildcard, self._cur_span())
    }

    fn _parse_class_pattern(self, name: str, start_index: int) -> Node {
        let open_index = self.current
        self._advance()  # `{`
        var fields: list[str?] = []
        var subs: list[Node] = []
        self._skip_newlines()
        if not self._check(TokenKind.RBrace) {
            loop {
                self._skip_newlines()
                let field_token = self._peek()
                let field = self._expect_ident_name("a field name in a class pattern")
                if field == null { break }
                var sub = _pat_bind(field_token.span, field)
                if self._match(TokenKind.Colon) { sub = self._parse_pattern() }
                fields.push(field)
                subs.push(sub)
                self._skip_newlines()
                if not self._match(TokenKind.Comma) { break }
                self._skip_newlines()
                if self._check(TokenKind.RBrace) or self._check(TokenKind.Eof) { break }
            }
        }
        self._close_sequence(TokenKind.RBrace, "}", "class pattern", open_index)
        return _pat_class(
            NodeKind.PatClass,
            self._span_since(start_index),
            name,
            null,
            fields,
            subs
        )
    }

    fn _parse_pattern(self) -> Node {
        if not self._enter_recursion() {
            self.expr_depth -= 1
            return _node(NodeKind.PatWildcard, self._cur_span())
        }
        defer { self.expr_depth -= 1 }

        let start_index = self.current
        let first = self._parse_pattern_primary()
        if not self._check(TokenKind.Pipe) { return first }

        var alternatives: list[Node] = [first]
        while self._match(TokenKind.Pipe) {
            alternatives.push(self._parse_pattern_primary())
            if self.panic_mode or self._check(TokenKind.Eof) { break }
        }
        return _pat_seq(NodeKind.PatOr, self._span_since(start_index), alternatives, -1)
    }

    # Reinterpret an already-parsed expression as an assignment pattern, for
    # `(a, b) = point`. Returns null when the expression is not a pattern.
    fn _expr_to_pattern(self, expr: Node) -> Node? {
        if expr.kind == NodeKind.Ident { return _pat_bind(expr.span, expr.text("name")) }
        if expr.kind != NodeKind.TupleLit and expr.kind != NodeKind.ListLit { return null }

        var elems: list[Node] = []
        for item in expr.child_list("items") {
            let sub = self._expr_to_pattern(item)
            if sub == null { return null }
            elems.push(sub)
        }
        let kind = expr.kind == NodeKind.TupleLit ? NodeKind.PatTuple : NodeKind.PatList
        return _pat_seq(kind, expr.span, elems, -1)
    }

    # --- blocks and statements -------------------------------------------
    fn _parse_block(self) -> Node {
        let start_index = self.current
        self._expect(TokenKind.LBrace, "`{` to open a block")

        var stmts: list[Node] = []
        self._skip_newlines()
        while not self._check(TokenKind.RBrace) and not self._check(TokenKind.Eof) {
            let before = self.current
            let stmt = self._parse_declaration()
            if stmt != null { stmts.push(stmt) }
            if self.panic_mode { self._synchronize() }
            if self.current == before { self._advance() }  # guarantee progress
            self._skip_newlines()
        }
        self._close_sequence(TokenKind.RBrace, "}", "block", start_index)
        return _node(NodeKind.Block, self._span_since(start_index), {"stmts": stmts})
    }

    fn _empty_block(self) -> Node {
        return _node(NodeKind.Block, self._cur_span(), {"stmts": []})
    }

    fn _expect_block(self, what: str) -> Node {
        if self._check(TokenKind.LBrace) { return self._parse_block() }

        # Jaithon 2 wrote `if c then do ... end`.
        if self._ident_text(0, "then") or self._ident_text(0, "do") {
            self._error_at(
                self._cur_span(),
                Code.E0104_EXPECTED_BLOCK,
                f"`{self._peek().text}` is not a Jaithon 3 keyword",
                help: f"blocks are braces: `{what} {{ ... }}`"
            )
            return self._empty_block()
        }

        self._error_at(
            self._cur_span(),
            Code.E0104_EXPECTED_BLOCK,
            f"expected `{{` to open the {what} body, " + f"found {self._describe(self._peek())}"
        )
        return self._empty_block()
    }

    fn _parse_if(self, as_expr: bool) -> Node {
        let start_index = self.current
        self._advance()  # `if` or `elif`
        let cond = self._parse_expression()
        let then_branch = self._expect_block("if")
        var else_branch: Node? = null

        if self._check(TokenKind.KwElif) {
            else_branch = self._parse_if(as_expr)
        } elif self._match(TokenKind.KwElse) {
            if self._check(TokenKind.KwIf) {
                else_branch = self._parse_if(as_expr)
            } else {
                else_branch = self._expect_block("else")
            }
        }

        let kind = as_expr ? NodeKind.IfExpr : NodeKind.If
        return _node(
            kind,
            self._span_since(start_index),
            {"cond": cond, "thenBranch": then_branch, "elseBranch": else_branch}
        )
    }

    fn _parse_while(self, label: str?, start_index: int) -> Node {
        self._advance()  # `while`
        let cond = self._parse_expression()
        let body = self._expect_block("while")

        return _node(
            NodeKind.While,
            self._span_since(start_index),
            {"cond": cond, "body": body, "label": label}
        )
    }

    fn _parse_loop(self, label: str?, start_index: int) -> Node {
        self._advance()  # `loop`
        let body = self._expect_block("loop")

        return _node(
            NodeKind.Loop,
            self._span_since(start_index),
            {"cond": null, "body": body, "label": label}
        )
    }

    fn _parse_for(self, label: str?, start_index: int) -> Node {
        self._advance()  # `for`
        let pattern = self._parse_pattern()
        self._expect(TokenKind.KwIn, "`in` after the loop pattern")
        let iterable = self._parse_expression()
        let body = self._expect_block("for")

        return _node(
            NodeKind.For,
            self._span_since(start_index),
            {"pattern": pattern, "iterable": iterable, "body": body, "label": label}
        )
    }

    fn _parse_match(self, as_expr: bool) -> Node {
        let start_index = self.current
        self._advance()  # `match`
        let subject = self._parse_expression()
        let kind = as_expr ? NodeKind.MatchExpr : NodeKind.Match

        var arms: list[dict[str, any]] = []
        let open_index = self.current
        if not self._expect(TokenKind.LBrace, "`{` to open the match body") {
            return _match_node(kind, self._span_since(start_index), subject, arms)
        }

        self._skip_newlines()
        while not self._check(TokenKind.RBrace) and not self._check(TokenKind.Eof) {
            let before = self.current
            let pattern = self._parse_pattern()
            var guard: Node? = null
            if self._match(TokenKind.KwIf) { guard = self._parse_expression() }
            self._expect(TokenKind.FatArrow, "`=>` after a match pattern")
            # In arm position a `{` opens a block, never a dict literal.
            let is_block = self._check(TokenKind.LBrace)
            let body = is_block ? self._parse_block() : self._parse_expression()
            arms.push(_arm(pattern, guard, body, self._span_since(before)))

            if self.panic_mode {
                self._synchronize()
            } else {
                self._match(TokenKind.Comma)
            }
            if self.current == before { self._advance() }
            self._skip_newlines()
        }
        self._close_sequence(TokenKind.RBrace, "}", "match body", open_index)
        return _match_node(kind, self._span_since(start_index), subject, arms)
    }

    fn _parse_try(self) -> Node {
        let start_index = self.current
        self._advance()  # `try`
        let body = self._expect_block("try")

        var catches: list[dict[str, any]] = []
        while self._check(TokenKind.KwCatch) {
            let catch_start = self.current
            self._advance()
            var name: str? = null
            var types: list[TypeExpr] = []
            if self._check(TokenKind.Ident) or self._check(TokenKind.Underscore) {
                name = self._expect_ident_name("an exception binding name")
            }
            if self._match(TokenKind.Colon) { types = self._parse_type_list() }
            let handler_body = self._expect_block("catch")
            catches.push(_catch(name, types, handler_body, self._span_since(catch_start)))
            if self.panic_mode { break }
        }

        var finally_block: Node? = null
        if self._match(TokenKind.KwFinally) { finally_block = self._expect_block("finally") }

        if catches.len() == 0 and finally_block == null {
            self._error_at_cur(
                Code.E0101_EXPECTED_TOKEN,
                "a `try` block needs at least one `catch` or a `finally`"
            )
        }

        return _node(
            NodeKind.Try,
            self._span_since(start_index),
            {"body": body, "catches": catches, "finallyBlock": finally_block}
        )
    }

    # True when the statement starts with a destructuring target: a bracketed
    # pattern whose matching close bracket is followed by `=`.
    fn _looks_like_destructuring(self) -> bool {
        let open = self._kind()
        if open != TokenKind.LParen and open != TokenKind.LBracket { return false }
        let closer = open == TokenKind.LParen ? TokenKind.RParen : TokenKind.RBracket

        var depth = 0
        var index = self.current
        loop {
            let kind = self._token_at(index).kind
            if kind == TokenKind.Eof { return false }
            if kind == TokenKind.LParen or kind == TokenKind.LBracket or kind == TokenKind.LBrace {
                depth += 1
            } elif kind == TokenKind.RParen or
                kind == TokenKind.RBracket or
                kind == TokenKind.RBrace {
                depth -= 1
                if depth == 0 {
                    if kind != closer { return false }
                    return self._token_at(index + 1).kind == TokenKind.Eq
                }
                if depth < 0 { return false }
            }
            index += 1
        }
    }

    fn _parse_expr_or_assign(self) -> Node {
        let start_index = self.current
        let destructuring = self._looks_like_destructuring()
        var target = destructuring ? self._parse_pattern() : self._parse_expression()

        let compound_op = _assign_op(self._kind())
        let compound = compound_op != null
        if not compound and not self._check(TokenKind.Eq) {
            self._end_statement()
            return _node(NodeKind.ExprStmt, self._span_since(start_index), {"expr": target})
        }

        let op_span = self._cur_span()
        self._advance()  # the assignment operator

        # `(a, b) = point` and `[x, ...rest] = xs` destructure.
        if not compound and (target.kind == NodeKind.TupleLit or target.kind == NodeKind.ListLit) {
            let as_pattern = self._expr_to_pattern(target)
            if as_pattern != null { target = as_pattern }
        }

        if not _is_assign_target(target) {
            let diag = self._error_at(
                target.span,
                Code.E0109_INVALID_ASSIGN_TARGET,
                "this expression cannot be assigned to"
            )
            if diag != null {
                diag.label(op_span, "assignment happens here")
                if target.kind == NodeKind.Call {
                    diag.with_help("a call result is a value; assign to a name, field, or index")
                }
            }
        }

        let value = self._parse_expression()
        self._end_statement()
        return _node(
            NodeKind.Assign,
            self._span_since(start_index),
            {
                "target": target,
                "value": value,
                "op": compound_op ?? OpKind.Add,
                "isCompound": compound,
            }
        )
    }

    # `print x`, `input name` and `system "ls"` were statements in Jaithon 2.
    fn _parse_legacy_call_statement(self, name: str) -> Node {
        let start_index = self.current
        let name_token = self._advance()
        let diag = self._error_recovered(
            name_token.span,
            Code.E0100_UNEXPECTED_TOKEN,
            f"`{name}` is a function in Jaithon 3"
        )

        var args: list[dict[str, any]] = []
        let arg_start = self.current
        loop {
            let one = self.current
            let value = self._parse_expression()
            args.push(_arg(null, value, false, self._span_since(one)))
            if not self._match(TokenKind.Comma) { break }
            if not self._starts_expression() or self.panic_mode { break }
        }
        if diag != null {
            let written = self.lexer.text_of(self._span_since(arg_start))
            diag.with_help(f"write `{name}({written})`")
        }

        let call = _node(
            NodeKind.Call,
            self._span_since(start_index),
            {"callee": _ident(name_token.span, name_token.text), "args": args}
        )
        let stmt = _node(NodeKind.ExprStmt, call.span, {"expr": call})
        self._end_statement()
        return stmt
    }

    # Diagnostics for the Jaithon 2 spellings that are now plain identifiers.
    # Returns `(handled, node)`: a handled statement may still be null, because
    # some of these recover by reporting and skipping.
    fn _check_legacy_statement(self) -> tuple[bool, Node?] {
        if not self._check(TokenKind.Ident) { return (false, null) }
        let text = self._peek().text

        if text == "print" or text == "input" or text == "system" {
            if not _starts_bare_argument(self._kind_at(1)) { return (false, null) }
            return (true, self._parse_legacy_call_statement(text))
        }

        # `end` is a plain identifier in Jaithon 3 (§1.5 does not reserve it),
        # so `end += 1` and `end = x` are statements, not stray block
        # terminators. Only the Jaithon 2 spelling — `end` alone on its line —
        # is diagnosed.
        if text == "end" {
            let next = self._kind_at(1)
            if next != TokenKind.Newline and next != TokenKind.Eof and next != TokenKind.RBrace {
                return (false, null)
            }
            self._error_at(
                self._cur_span(),
                Code.E0100_UNEXPECTED_TOKEN,
                "`end` is not a Jaithon 3 keyword",
                help: "blocks close with `}`"
            )
            self._advance()
            return (true, null)
        }

        if (text == "func" or text == "def") and self._kind_at(1) == TokenKind.Ident {
            self._error_recovered(
                self._cur_span(),
                Code.E0100_UNEXPECTED_TOKEN,
                f"`{text}` is not a Jaithon 3 keyword",
                help: "functions are declared with `fn`"
            )
            return (true, self._parse_fn_rest(Visibility.Private, false, self.current, true, false))
        }

        return (false, null)
    }

    fn _parse_statement(self) -> Node? {
        let start_index = self.current
        let kind = self._kind()

        if kind == TokenKind.LBrace { return self._parse_block() }
        if kind == TokenKind.KwIf { return self._parse_if(false) }
        if kind == TokenKind.KwWhile { return self._parse_while(null, start_index) }
        if kind == TokenKind.KwLoop { return self._parse_loop(null, start_index) }
        if kind == TokenKind.KwFor { return self._parse_for(null, start_index) }
        if kind == TokenKind.KwMatch { return self._parse_match(false) }
        if kind == TokenKind.KwTry { return self._parse_try() }

        if kind == TokenKind.Label {
            let label = self._label_name(self._advance())
            self._expect(TokenKind.Colon, "`:` after a loop label")
            if self._check(TokenKind.KwWhile) { return self._parse_while(label, start_index) }
            if self._check(TokenKind.KwLoop) { return self._parse_loop(label, start_index) }
            if self._check(TokenKind.KwFor) { return self._parse_for(label, start_index) }
            self._error_at(
                self._span_since(start_index),
                Code.E0115_LABEL_NOT_ON_LOOP,
                f"`'{label}` labels something that is not a loop",
                help: "only `while`, `loop`, and `for` may be labelled"
            )
            return null
        }

        if kind == TokenKind.KwReturn {
            self._advance()
            let value = self._starts_expression() ? self._parse_expression() : null
            self._end_statement()
            return _node(NodeKind.Return, self._span_since(start_index), {"value": value})
        }
        if kind == TokenKind.KwThrow {
            self._advance()
            let value = self._parse_expression()
            self._end_statement()
            return _node(NodeKind.Throw, self._span_since(start_index), {"value": value})
        }
        if kind == TokenKind.KwBreak or kind == TokenKind.KwContinue {
            self._advance()
            var label: str? = null
            if self._check(TokenKind.Label) { label = self._label_name(self._advance()) }
            self._end_statement()
            let jump = kind == TokenKind.KwBreak ? NodeKind.Break : NodeKind.Continue
            return _node(jump, self._span_since(start_index), {"label": label})
        }
        if kind == TokenKind.KwDefer {
            self._advance()
            let body = self._expect_block("defer")
            return _node(NodeKind.Defer, self._span_since(start_index), {"body": body})
        }
        if kind == TokenKind.KwAssert {
            self._advance()
            let cond = self._parse_expression()
            let message = self._match(TokenKind.Comma) ? self._parse_expression() : null
            self._end_statement()
            return _node(
                NodeKind.Assert,
                self._span_since(start_index),
                {"cond": cond, "message": message}
            )
        }

        if kind == TokenKind.KwElse or
            kind == TokenKind.KwElif or
            kind == TokenKind.KwCatch or
            kind == TokenKind.KwFinally {
            let word = self._peek().text
            self._error_at(
                self._cur_span(),
                Code.E0100_UNEXPECTED_TOKEN,
                f"`{word}` has no matching statement before it",
                help: f"put `{word}` on the same line as the closing `}}`"
            )
            return null
        }

        if kind == TokenKind.KwImpl {
            self._error_at(
                self._cur_span(),
                Code.E0100_UNEXPECTED_TOKEN,
                "`impl` blocks are not part of Jaithon 3",
                help: "declare methods inside the class body and list traits with `:`"
            )
            return null
        }

        if kind == TokenKind.Ident {
            let (handled, legacy) = self._check_legacy_statement()
            if handled { return legacy }
        }

        return self._parse_expr_or_assign()
    }

    # --- declarations ----------------------------------------------------

    # §1.4: `__name__` is reserved for the implementation.
    fn _check_reserved_name(self, name: str?, span: Span) -> void {
        if name == null or not _is_reserved_name(name) { return }
        self._error_recovered(
            span,
            Code.E0114_RESERVED_IDENTIFIER,
            f"`{name}` is reserved for the implementation",
            help: "names of the form `__x__` may not be declared here"
        )
    }

    # §7.2: `Self` is in scope in every class, trait and enum body as a name for
    # the implementing type, so nothing may declare a type of that name — a
    # `class Self` would be reachable only from outside its own body.
    fn _check_reserved_type_name(self, name: str?, span: Span) -> void {
        if name != "Self" { return }
        self._error_recovered(
            span,
            Code.E0114_RESERVED_IDENTIFIER,
            "`Self` is reserved: it names the implementing type",
            help: "`Self` is already in scope inside every class, trait and enum body"
        )
    }

    fn _check_duplicate_param(self, params: list[dict[str, any]], name: str, span: Span) -> void {
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

    fn _parse_generics(self) -> list[dict[str, any]] {
        var generics: list[dict[str, any]] = []
        if not self._check(TokenKind.LBracket) { return generics }

        let open_index = self.current
        self._advance()
        if not self._check(TokenKind.RBracket) {
            loop {
                let name_token = self._peek()
                let name = self._expect_ident_name("a type parameter name")
                if name == null { break }
                var bound: TypeExpr? = null
                if self._match(TokenKind.Colon) { bound = self._parse_type() }
                self._check_reserved_type_name(name, name_token.span)
                generics.push(_generic(name, bound, name_token.span))
                if not self._match(TokenKind.Comma) { break }
                if self._check(TokenKind.RBracket) or self._check(TokenKind.Eof) { break }
            }
        }
        self._close_sequence(TokenKind.RBracket, "]", "type parameter list", open_index)
        return generics
    }

    # Parse a parameter list up to and including the closing `)`.
    fn _parse_params(self, open_index: int) -> list[dict[str, any]] {
        var params: list[dict[str, any]] = []
        var rest_span: Span? = null
        var default_span: Span? = null

        self._skip_newlines()
        if not self._check(TokenKind.RParen) {
            loop {
                self._skip_newlines()
                let param_start = self.current
                let is_variadic = self._match(TokenKind.Ellipsis)
                let is_kw_rest = not is_variadic and self._match(TokenKind.StarStar)

                var name: str? = null
                if self._check(TokenKind.KwSelf) {
                    self._advance()
                    name = "self"
                } else {
                    name = self._expect_ident_name("a parameter name")
                    if name == null { break }
                }

                var declared: TypeExpr? = null
                var default_value: Node? = null
                if self._match(TokenKind.Colon) { declared = self._parse_type() }
                if self._match(TokenKind.Eq) { default_value = self._parse_expression() }
                let span = self._span_since(param_start)

                self._check_duplicate_param(params, name ?? "_", span)
                if rest_span != null {
                    let diag = self._error_recovered(
                        span,
                        Code.E0112_PARAM_AFTER_VARIADIC,
                        f"parameter `{name}` cannot follow a " + "rest parameter"
                    )
                    if diag != null { diag.label(rest_span, "rest parameter declared here") }
                } elif default_span != null and
                    default_value == null and
                    not is_variadic and
                    not is_kw_rest {
                    let diag = self._error_recovered(
                        span,
                        Code.E0113_DEFAULT_BEFORE_REQUIRED,
                        f"required parameter `{name}` follows " + "a defaulted parameter"
                    )
                    if diag != null {
                        diag.label(default_span, "this parameter has a default value")
                    }
                }
                if is_variadic or is_kw_rest { rest_span = span }
                if default_value != null { default_span = span }

                params.push(_param(name, declared, default_value, span, is_variadic, is_kw_rest))
                self._skip_newlines()
                if not self._match(TokenKind.Comma) { break }
                self._skip_newlines()
                if self._check(TokenKind.RParen) or self._check(TokenKind.Eof) or self.panic_mode {
                    break
                }
            }
        }
        self._close_sequence(TokenKind.RParen, ")", "parameter list", open_index)
        return params
    }

    # Shared by `fn` declarations, methods, trait members and anonymous `fn`.
    # The cursor sits on the `fn` keyword, or on the Jaithon 2 spelling of it.
    fn _parse_fn_rest(
        self,
        visibility: Visibility,
        is_static: bool,
        start_index: int,
        named: bool,
        body_optional: bool
    ) -> Node? {
        self._advance()  # `fn`

        var name: str? = null
        if named {
            name = self._expect_ident_name("a function name")
            if name == null { return null }
        }

        let generics = self._parse_generics()
        var params: list[dict[str, any]] = []
        let open_index = self.current
        if self._expect(TokenKind.LParen, "`(` to open the parameter list") {
            params = self._parse_params(open_index)
        }
        var return_type: TypeExpr? = null
        if self._match(TokenKind.Arrow) { return_type = self._parse_type() }

        let saved_yield = self.yield_count
        self.yield_count = 0
        var body: Node? = null
        var is_expr_body = false

        if body_optional and not self._check(TokenKind.LBrace) {
            body = null  # a trait requirement: a signature and no definition
        } elif self._check(TokenKind.FatArrow) {
            # Not in the grammar, but a common typo carried over from lambdas.
            self._error_at(
                self._cur_span(),
                Code.E0104_EXPECTED_BLOCK,
                "a function body is a block, not `=>`",
                help: "write `{ return <expr> }`, or use a lambda `|x| <expr>`"
            )
            self._advance()
            body = self._parse_expression()
            is_expr_body = true
        } else {
            body = self._expect_block("function")
        }

        let is_generator = self.yield_count > 0
        self.yield_count = saved_yield

        return _node(
            named ? NodeKind.FnDecl : NodeKind.AnonFn,
            self._span_since(start_index),
            {
                "name": name,
                "generics": generics,
                "params": params,
                "returnType": return_type,
                "body": body,
                "visibility": visibility,
                "isStatic": is_static,
                "isExprBody": is_expr_body,
                "isGenerator": is_generator,
                "isAsync": false,
            }
        )
    }

    # `get area() -> float { ... }` and `set width(v: float) { ... }`.
    fn _parse_accessor(self, visibility: Visibility, is_static: bool, is_setter: bool) -> Node? {
        let start_index = self.current
        self._advance()  # `get` or `set`
        let name = self._expect_ident_name("a property name")
        if name == null { return null }

        var params: list[dict[str, any]] = []
        let open_index = self.current
        if self._expect(TokenKind.LParen, "`(` after the property name") {
            params = self._parse_params(open_index)
        }
        var return_type: TypeExpr? = null
        if self._match(TokenKind.Arrow) { return_type = self._parse_type() }

        let saved_yield = self.yield_count
        self.yield_count = 0
        let body = self._expect_block(is_setter ? "set" : "get")
        self.yield_count = saved_yield

        return _node(
            NodeKind.FnDecl,
            self._span_since(start_index),
            {
                "name": name,
                "generics": [],
                "params": params,
                "returnType": return_type,
                "body": body,
                "visibility": visibility,
                "isStatic": is_static,
                "isExprBody": false,
                "isGenerator": false,
                "isAsync": false,
            }
        )
    }

    fn _parse_var_decl(self, visibility: Visibility, start_index: int) -> Node {
        let keyword = self._kind()
        var decl_kind = DeclKind.Const
        if keyword == TokenKind.KwLet {
            decl_kind = DeclKind.Let
        } elif keyword == TokenKind.KwVar {
            decl_kind = DeclKind.Var
        }
        self._advance()

        let pattern = self._parse_pattern()
        if pattern.kind == NodeKind.PatBind {
            self._check_reserved_name(pattern.text("name"), pattern.span)
        }
        var declared: TypeExpr? = null
        if self._match(TokenKind.Colon) { declared = self._parse_type() }

        var init: Node? = null
        if self._match(TokenKind.Eq) {
            init = self._parse_expression()
        } elif decl_kind != DeclKind.Var and not self.panic_mode {
            let keyword_text = decl_kind == DeclKind.Let ? "let" : "const"
            self._error_at(
                self._span_since(start_index),
                Code.E0304_MISSING_INITIALIZER,
                f"`{keyword_text}` bindings must be initialised",
                help: "add `= <expression>`, or use `var` to declare it uninitialised"
            )
        }
        self._end_statement()

        return _node(
            NodeKind.VarDecl,
            self._span_since(start_index),
            {
                "varDeclKind": decl_kind,
                "pattern": pattern,
                "declaredType": declared,
                "init": init,
                "visibility": visibility,
            }
        )
    }

    fn _parse_class_decl(self, visibility: Visibility, start_index: int) -> Node? {
        self._advance()  # `class`
        let name_token = self._peek()
        let name = self._expect_ident_name("a class name")
        if name == null { return null }
        self._check_reserved_type_name(name, name_token.span)

        let generics = self._parse_generics()
        var superclass: TypeExpr? = null
        if self._match(TokenKind.KwExtends) { superclass = self._parse_type() }
        var traits: list[TypeExpr] = []
        if self._match(TokenKind.Colon) { traits = self._parse_type_list() }

        var fields: list[dict[str, any]] = []
        var methods: list[Node] = []
        var getters: list[Node] = []
        var setters: list[Node] = []

        let open_index = self.current
        if self._expect(TokenKind.LBrace, "`{` to open the class body") {
            self._skip_newlines()
            while not self._check(TokenKind.RBrace) and not self._check(TokenKind.Eof) {
                let before = self.current
                let member_visibility = self._parse_member_visibility()
                let member_static = self._match(TokenKind.KwStatic)

                if self._check(TokenKind.KwVar) or self._check(TokenKind.KwLet) {
                    let field = self._parse_field(member_visibility, member_static)
                    if field != null { fields.push(field) }
                } elif self._check(TokenKind.KwConst) {
                    self._error_at(
                        self._cur_span(),
                        Code.E0100_UNEXPECTED_TOKEN,
                        "class fields are declared with `var` or `let`",
                        help: "write `static let NAME: <type> = <value>`"
                    )
                } elif self._check(TokenKind.KwFn) {
                    let method = self._parse_fn_rest(
                        member_visibility,
                        member_static,
                        self.current,
                        true,
                        false
                    )
                    if method != null { methods.push(method) }
                } elif self._ident_text(0, "get") and self._kind_at(1) == TokenKind.Ident {
                    let getter = self._parse_accessor(member_visibility, member_static, false)
                    if getter != null { getters.push(getter) }
                } elif self._ident_text(0, "set") and self._kind_at(1) == TokenKind.Ident {
                    let setter = self._parse_accessor(member_visibility, member_static, true)
                    if setter != null { setters.push(setter) }
                } else {
                    self._error_at(
                        self._cur_span(),
                        Code.E0103_EXPECTED_STATEMENT,
                        f"expected a class member, " + f"found {self._describe(self._peek())}",
                        help: "members are `var`/`let` fields, `fn` methods, " +
                            "or `get`/`set` properties"
                    )
                }

                if self.panic_mode { self._synchronize() }
                if self.current == before { self._advance() }
                self._skip_newlines()
            }
            self._close_sequence(TokenKind.RBrace, "}", "class body", open_index)
        }

        return _node(
            NodeKind.ClassDecl,
            self._span_since(start_index),
            {
                "name": name,
                "generics": generics,
                "superclass": superclass,
                "traits": traits,
                "fields": fields,
                "methods": methods,
                "getters": getters,
                "setters": setters,
                "visibility": visibility,
                "isAbstract": false,
            }
        )
    }

    fn _parse_member_visibility(self) -> Visibility {
        if self._match(TokenKind.KwPub) { return Visibility.Public }
        if self._match(TokenKind.KwProt) { return Visibility.Protected }
        return Visibility.Private
    }

    fn _parse_field(self, visibility: Visibility, is_static: bool) -> dict[str, any]? {
        let field_start = self.current
        let is_let = self._check(TokenKind.KwLet)
        self._advance()

        let name_token = self._peek()
        let name = self._expect_ident_name("a field name")
        if name == null {
            self._end_statement()
            return null
        }

        var declared: TypeExpr? = null
        if self._match(TokenKind.Colon) {
            declared = self._parse_type()
        } else {
            let keyword_text = is_let ? "let" : "var"
            self._error_at(
                name_token.span,
                Code.E0105_EXPECTED_TYPE,
                f"field `{name}` needs a type annotation",
                help: f"write `{keyword_text} {name}: <type>`"
            )
        }
        var default_value: Node? = null
        if self._match(TokenKind.Eq) { default_value = self._parse_expression() }
        let span = self._span_since(field_start)
        self._end_statement()

        return _field(name, declared, default_value, visibility, is_static, is_let, span)
    }

    fn _parse_trait_decl(self, visibility: Visibility, start_index: int) -> Node? {
        self._advance()  # `trait`
        let name_token = self._peek()
        let name = self._expect_ident_name("a trait name")
        if name == null { return null }
        self._check_reserved_type_name(name, name_token.span)

        let generics = self._parse_generics()
        var supers: list[TypeExpr] = []
        if self._match(TokenKind.Colon) { supers = self._parse_type_list() }

        var methods: list[Node] = []
        let open_index = self.current
        if self._expect(TokenKind.LBrace, "`{` to open the trait body") {
            self._skip_newlines()
            while not self._check(TokenKind.RBrace) and not self._check(TokenKind.Eof) {
                let before = self.current
                let member_visibility = self._parse_member_visibility()
                let member_static = self._match(TokenKind.KwStatic)

                if self._check(TokenKind.KwFn) {
                    let method = self._parse_fn_rest(
                        member_visibility,
                        member_static,
                        self.current,
                        true,
                        true
                    )
                    if method != null {
                        methods.push(method)
                        if method.child("body") == null { self._end_statement() }
                    }
                } else {
                    self._error_at(
                        self._cur_span(),
                        Code.E0103_EXPECTED_STATEMENT,
                        f"expected `fn` in a trait body, " + f"found {self._describe(self._peek())}"
                    )
                }
                if self.panic_mode { self._synchronize() }
                if self.current == before { self._advance() }
                self._skip_newlines()
            }
            self._close_sequence(TokenKind.RBrace, "}", "trait body", open_index)
        }

        return _node(
            NodeKind.TraitDecl,
            self._span_since(start_index),
            {
                "name": name,
                "generics": generics,
                "supers": supers,
                "methods": methods,
                "visibility": visibility,
            }
        )
    }

    fn _parse_enum_decl(self, visibility: Visibility, start_index: int) -> Node? {
        self._advance()  # `enum`
        let name_token = self._peek()
        let name = self._expect_ident_name("an enum name")
        if name == null { return null }
        self._check_reserved_type_name(name, name_token.span)

        let generics = self._parse_generics()
        var variants: list[dict[str, any]] = []
        var methods: list[Node] = []

        let open_index = self.current
        if self._expect(TokenKind.LBrace, "`{` to open the enum body") {
            self._skip_newlines()
            while not self._check(TokenKind.RBrace) and not self._check(TokenKind.Eof) {
                let before = self.current
                if self._check(TokenKind.KwFn) or
                    self._check(TokenKind.KwPub) or
                    self._check(TokenKind.KwStatic) {
                    var member_visibility = Visibility.Private
                    if self._match(TokenKind.KwPub) { member_visibility = Visibility.Public }
                    let member_static = self._match(TokenKind.KwStatic)
                    if self._check(TokenKind.KwFn) {
                        let method = self._parse_fn_rest(
                            member_visibility,
                            member_static,
                            self.current,
                            true,
                            false
                        )
                        if method != null { methods.push(method) }
                    } else {
                        self._error_at_cur(
                            Code.E0103_EXPECTED_STATEMENT,
                            "expected `fn` after `pub`/`static`"
                        )
                    }
                } else {
                    let variant = self._parse_variant()
                    if variant != null { variants.push(variant) }
                    self._match(TokenKind.Comma)
                }
                if self.panic_mode { self._synchronize() }
                if self.current == before { self._advance() }
                self._skip_newlines()
            }
            self._close_sequence(TokenKind.RBrace, "}", "enum body", open_index)
        }

        return _node(
            NodeKind.EnumDecl,
            self._span_since(start_index),
            {
                "name": name,
                "generics": generics,
                "variants": variants,
                "methods": methods,
                "visibility": visibility,
            }
        )
    }

    fn _parse_variant(self) -> dict[str, any]? {
        let variant_start = self.current
        let name = self._expect_ident_name("a variant name")
        if name == null { return null }

        var params: list[dict[str, any]] = []
        if self._check(TokenKind.LParen) {
            let payload_index = self.current
            self._advance()
            params = self._parse_params(payload_index)
        }
        return _variant(name, params, self._span_since(variant_start))
    }

    fn _parse_type_decl(self, visibility: Visibility, start_index: int) -> Node? {
        self._advance()  # `type`
        let name_token = self._peek()
        let name = self._expect_ident_name("a type name")
        if name == null { return null }
        self._check_reserved_type_name(name, name_token.span)

        let generics = self._parse_generics()
        self._expect(TokenKind.Eq, "`=` in a type alias")
        let aliased = self._parse_type()
        self._end_statement()

        return _node(
            NodeKind.TypeDecl,
            self._span_since(start_index),
            {
                "name": name,
                "generics": generics,
                "aliased": aliased,
                "visibility": visibility,
            }
        )
    }

    # `std.math`, `.sibling`, `..pkg.mod`
    fn _parse_module_path(self) -> str? {
        var path = ""
        while self._check(TokenKind.Dot) or
            self._check(TokenKind.DotDot) or
            self._check(TokenKind.Ellipsis) {
            path = path + self._advance().text
        }

        if not self._check(TokenKind.Ident) {
            self._error_at(
                self._cur_span(),
                Code.E0804_INVALID_MODULE_PATH,
                f"expected a module path, found {self._describe(self._peek())}"
            )
            return null
        }
        path = path + self._advance().text
        while self._check(TokenKind.Dot) and self._kind_at(1) == TokenKind.Ident {
            self._advance()
            path = path + "." + self._advance().text
        }
        return path
    }

    fn _parse_import(self, start_index: int) -> Node? {
        self._advance()  # `import`
        let path = self._parse_module_path()
        if path == null { return null }
        var alias: str? = null
        if self._match(TokenKind.KwAs) { alias = self._expect_ident_name("an import alias") }
        self._end_statement()

        return _node(NodeKind.Import, self._span_since(start_index), {"path": path, "alias": alias})
    }

    fn _parse_from_import(self, start_index: int) -> Node? {
        self._advance()  # `from`
        let path = self._parse_module_path()
        if path == null { return null }
        self._expect(TokenKind.KwImport, "`import` after the module path")

        var wildcard = false
        var items: list[dict[str, any]] = []
        if self._match(TokenKind.Star) {
            wildcard = true
        } else {
            let parens = self._match(TokenKind.LParen)
            loop {
                self._skip_newlines()
                let name_token = self._peek()
                let name = self._expect_ident_name("an imported name")
                if name == null { break }
                var alias: str? = null
                if self._match(TokenKind.KwAs) {
                    alias = self._expect_ident_name("an import alias")
                }
                items.push(_import_item(name, alias, name_token.span))
                if not self._match(TokenKind.Comma) { break }
                if self._check(TokenKind.Eof) or self.panic_mode { break }
            }
            if parens {
                self._skip_newlines()
                self._expect(TokenKind.RParen, "`)` to close the import list")
            }
        }
        self._end_statement()

        return _node(
            NodeKind.FromImport,
            self._span_since(start_index),
            {"path": path, "items": items, "isWildcard": wildcard}
        )
    }

    fn _parse_export_decl(self, start_index: int) -> Node {
        self._advance()  # `export`
        var names: list[str] = []

        let open_index = self.current
        if self._expect(TokenKind.LBrace, "`{` to open the export list") {
            self._skip_newlines()
            if not self._check(TokenKind.RBrace) {
                loop {
                    self._skip_newlines()
                    let name = self._expect_ident_name("an exported name")
                    if name == null { break }
                    names.push(name)
                    self._skip_newlines()
                    if not self._match(TokenKind.Comma) { break }
                    self._skip_newlines()
                    if self._check(TokenKind.RBrace) or self._check(TokenKind.Eof) { break }
                }
            }
            self._skip_newlines()
            self._close_sequence(TokenKind.RBrace, "}", "export list", open_index)
        }
        self._end_statement()

        return _node(NodeKind.Export, self._span_since(start_index), {"names": names})
    }

    fn _parse_module_decl(self, start_index: int) -> Node? {
        self._advance()  # `module`
        let name = self._parse_module_path()
        if name == null { return null }
        self._end_statement()

        return _node(NodeKind.ModuleDecl, self._span_since(start_index), {"name": name})
    }

    fn _parse_declaration(self) -> Node? {
        let start_index = self.current
        var visibility = Visibility.Private
        var saw_visibility = false
        if self._match(TokenKind.KwPub) {
            visibility = Visibility.Public
            saw_visibility = true
        } elif self._match(TokenKind.KwProt) {
            visibility = Visibility.Protected
            saw_visibility = true
        }
        let is_static = self._match(TokenKind.KwStatic)

        let kind = self._kind()
        if kind == TokenKind.KwClass { return self._parse_class_decl(visibility, start_index) }
        if kind == TokenKind.KwTrait { return self._parse_trait_decl(visibility, start_index) }
        if kind == TokenKind.KwEnum { return self._parse_enum_decl(visibility, start_index) }
        if kind == TokenKind.KwType { return self._parse_type_decl(visibility, start_index) }
        if kind == TokenKind.KwLet or kind == TokenKind.KwVar or kind == TokenKind.KwConst {
            return self._parse_var_decl(visibility, start_index)
        }
        # `fn name(...)` declares; `fn(...)` is an anonymous function value.
        if kind == TokenKind.KwFn and self._kind_at(1) == TokenKind.Ident {
            return self._parse_fn_rest(visibility, is_static, start_index, true, false)
        }
        if kind == TokenKind.KwImport or
            kind == TokenKind.KwFrom or
            kind == TokenKind.KwExport or
            kind == TokenKind.KwModule {
            if saw_visibility or is_static {
                self._error_at(
                    self._span_since(start_index),
                    Code.E0100_UNEXPECTED_TOKEN,
                    "imports and exports take no visibility modifier",
                    help: "delete the modifier"
                )
                return null
            }
            if kind == TokenKind.KwImport { return self._parse_import(start_index) }
            if kind == TokenKind.KwFrom { return self._parse_from_import(start_index) }
            if kind == TokenKind.KwExport { return self._parse_export_decl(start_index) }
            return self._parse_module_decl(start_index)
        }

        if saw_visibility or is_static {
            var modifier = visibility == Visibility.Public ? "pub" : "prot"
            if is_static { modifier = "static" }
            self._error_at(
                self._span_since(start_index),
                Code.E0103_EXPECTED_STATEMENT,
                f"expected a declaration after `{modifier}`, " +
                    f"found {self._describe(self._peek())}",
                help: "`pub` may precede `fn`, `class`, `trait`, `enum`, `type`, " +
                    "`let`, `var`, or `const`"
            )
            return null
        }

        return self._parse_statement()
    }
}

#: Parse `source` into a `NodeKind.Program` node, or null when it had any error.
#:
#: Diagnostics land in `diagnostics`, which the caller supplies when it wants to
#: inspect them; the lexer and the parser share one bag.
#:
#: ```
#: let bag = DiagnosticBag()
#: parse("let x = \n", diagnostics: bag)     # null
#: bag.has_errors()                          # true
#: ```
pub fn parse(source: str, file_id: int = 0, diagnostics: DiagnosticBag? = null) -> Node? {
    let bag = diagnostics ?? DiagnosticBag()
    let lexer = Lexer(source, file_id, bag)
    let parser = Parser(lexer)
    let program = parser.parse_program()
    if lexer.had_error { return null }
    return program
}

# --- node construction -------------------------------------------------
#
# Every node this parser builds is created through these helpers, so the
# coupling to `jaithon.ast` stays narrow: a node is its kind, its span, and a
# payload keyed by the field names of the AST encoding (spec §12) — which is
# what makes the two front ends diffable as JSON. Sub-structures that are not
# nodes in their own right (parameters, arguments, match arms, and the rest)
# are records with those same keys.
fn _node(kind: NodeKind, span: Span, fields: dict[str, any] = {}) -> Node {
    return Node(kind, fields, span)
}

fn _type(
    kind: TypeKind,
    span: Span,
    name: str? = null,
    args: list[TypeExpr] = [],
    inner: TypeExpr? = null
) -> TypeExpr {
    return TypeExpr(kind, name, args, inner, span)
}

fn _join(first: Span, second: Span) -> Span {
    if first.file < 0 { return second }
    if second.file < 0 { return first }
    if first.file != second.file { return first }  # a cross-file join keeps the first
    let start = min(first.start, second.start)
    return Span(first.file, start, max(start, max(first.end, second.end)))
}

fn _ident(span: Span, name: str) -> Node {
    return _node(NodeKind.Ident, span, {"name": name})
}

fn _str_lit(span: Span, chars: str) -> Node {
    return _node(NodeKind.StrLit, span, {"chars": chars})
}

fn _sequence(kind: NodeKind, span: Span, items: list[Node]) -> Node {
    return _node(kind, span, {"items": items})
}

fn _dict_lit(span: Span, keys: list[Node], values: list[Node]) -> Node {
    return _node(NodeKind.DictLit, span, {"keys": keys, "values": values})
}

fn _unary(span: Span, op: OpKind, operand: Node) -> Node {
    return _node(NodeKind.Unary, span, {"op": op, "operand": operand})
}

fn _binary(kind: NodeKind, op: OpKind, left: Node, right: Node) -> Node {
    return _node(kind, _join(left.span, right.span), {"op": op, "left": left, "right": right})
}

fn _wrap(kind: NodeKind, span: Span, operand: Node) -> Node {
    return _node(kind, span, {"operand": operand})
}

fn _member(kind: NodeKind, span: Span, object: Node, name: str) -> Node {
    return _node(kind, span, {"object": object, "name": name})
}

fn _match_node(kind: NodeKind, span: Span, subject: Node, arms: list[dict[str, any]]) -> Node {
    return _node(kind, span, {"subject": subject, "arms": arms})
}

fn _pat_bind(span: Span, name: str?) -> Node {
    return _node(NodeKind.PatBind, span, {"name": name, "type": null})
}

fn _pat_seq(kind: NodeKind, span: Span, elems: list[Node], rest_index: int) -> Node {
    return _node(kind, span, {"elems": elems, "restIndex": rest_index})
}

fn _pat_class(
    kind: NodeKind,
    span: Span,
    type_name: str,
    variant_name: str?,
    field_names: list[str?],
    sub_patterns: list[Node]
) -> Node {
    return _node(
        kind,
        span,
        {
            "typeName": type_name,
            "variantName": variant_name,
            "fieldNames": field_names,
            "subPatterns": sub_patterns,
        }
    )
}

# A positional class or enum pattern still carries one field-name slot per
# sub-pattern; a null entry means "matched by position".
fn _nulls(count: int) -> list[str?] {
    return [null for _index in 0..count]
}

fn _param(
    name: str?,
    declared: TypeExpr?,
    default_value: Node?,
    span: Span,
    is_variadic: bool,
    is_kw_rest: bool
) -> dict[str, any] {
    return {
        "name": name,
        "type": declared,
        "defaultValue": default_value,
        "span": span,
        "isVariadic": is_variadic,
        "isKwRest": is_kw_rest,
    }
}

fn _arg(name: str?, value: Node, is_spread: bool, span: Span) -> dict[str, any] {
    return {"name": name, "value": value, "isSpread": is_spread, "span": span}
}

fn _arm(pattern: Node, guard: Node?, body: Node, span: Span) -> dict[str, any] {
    return {"pattern": pattern, "guard": guard, "body": body, "span": span}
}

fn _comp_clause(
    pattern: Node,
    iterable: Node,
    conditions: list[Node],
    span: Span
) -> dict[str, any] {
    return {"pattern": pattern, "iterable": iterable, "conditions": conditions, "span": span}
}

fn _catch(name: str?, types: list[TypeExpr], body: Node, span: Span) -> dict[str, any] {
    return {"name": name, "types": types, "body": body, "span": span}
}

fn _field(
    name: str?,
    declared: TypeExpr?,
    default_value: Node?,
    visibility: Visibility,
    is_static: bool,
    is_let: bool,
    span: Span
) -> dict[str, any] {
    return {
        "name": name,
        "type": declared,
        "defaultValue": default_value,
        "visibility": visibility,
        "isStatic": is_static,
        "isLet": is_let,
        "span": span,
    }
}

fn _variant(name: str?, params: list[dict[str, any]], span: Span) -> dict[str, any] {
    return {"name": name, "params": params, "span": span}
}

fn _generic(name: str?, bound: TypeExpr?, span: Span) -> dict[str, any] {
    return {"name": name, "bound": bound, "span": span}
}

fn _import_item(name: str?, alias: str?, span: Span) -> dict[str, any] {
    return {"name": name, "alias": alias, "span": span}
}

# --- token classification ----------------------------------------------
#
# Each operator table returns null for a token that is not an operator of that
# level, which is what ends the corresponding loop in the ladder.
fn _multiplicative_op(kind: TokenKind) -> OpKind? {
    return match kind {
        TokenKind.Star => OpKind.Mul,
        TokenKind.Slash => OpKind.Div,
        TokenKind.SlashSlash => OpKind.FloorDiv,
        TokenKind.Percent => OpKind.Mod,
        TokenKind.At => OpKind.MatMul,
        TokenKind.StarPercent => OpKind.MulWrap,
        _ => null,
    }
}

fn _additive_op(kind: TokenKind) -> OpKind? {
    return match kind {
        TokenKind.Plus => OpKind.Add,
        TokenKind.Minus => OpKind.Sub,
        TokenKind.PlusPercent => OpKind.AddWrap,
        TokenKind.MinusPercent => OpKind.SubWrap,
        _ => null,
    }
}

fn _compare_op(kind: TokenKind) -> OpKind? {
    return match kind {
        TokenKind.Lt => OpKind.Lt,
        TokenKind.Le => OpKind.Le,
        TokenKind.Gt => OpKind.Gt,
        TokenKind.Ge => OpKind.Ge,
        TokenKind.EqEq => OpKind.Eq,
        TokenKind.BangEq => OpKind.Ne,
        TokenKind.KwIn => OpKind.In,
        _ => null,
    }
}

fn _assign_op(kind: TokenKind) -> OpKind? {
    return match kind {
        TokenKind.PlusEq => OpKind.Add,
        TokenKind.MinusEq => OpKind.Sub,
        TokenKind.StarEq => OpKind.Mul,
        TokenKind.SlashEq => OpKind.Div,
        TokenKind.SlashSlashEq => OpKind.FloorDiv,
        TokenKind.PercentEq => OpKind.Mod,
        TokenKind.StarStarEq => OpKind.Pow,
        TokenKind.AmpEq => OpKind.Band,
        TokenKind.PipeEq => OpKind.Bor,
        TokenKind.CaretEq => OpKind.Bxor,
        TokenKind.ShlEq => OpKind.Shl,
        TokenKind.ShrEq => OpKind.Shr,
        _ => null,
    }
}

fn _starts_expression_kind(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.Int | TokenKind.Float | TokenKind.Str | TokenKind.FStringStart => true,
        TokenKind.Ident | TokenKind.Underscore => true,
        TokenKind.LParen | TokenKind.LBracket | TokenKind.LBrace => true,
        TokenKind.Minus | TokenKind.Plus | TokenKind.Tilde | TokenKind.Pipe => true,
        TokenKind.KwNot | TokenKind.KwTrue | TokenKind.KwFalse | TokenKind.KwNull => true,
        TokenKind.KwSelf | TokenKind.KwSuper | TokenKind.KwFn | TokenKind.KwIf => true,
        TokenKind.KwMatch | TokenKind.KwYield => true,
        # `throw e` in expression position has type `never` (§4.3). The
        # statement form is matched earlier, so this only affects places that
        # genuinely expect an expression.
        TokenKind.KwThrow => true,
        _ => false,
    }
}

fn _starts_declaration(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.KwImport | TokenKind.KwFrom | TokenKind.KwExport | TokenKind.KwModule => true,
        TokenKind.KwClass | TokenKind.KwTrait | TokenKind.KwEnum | TokenKind.KwType => true,
        TokenKind.KwFn | TokenKind.KwLet | TokenKind.KwVar | TokenKind.KwConst => true,
        TokenKind.KwPub | TokenKind.KwProt | TokenKind.KwStatic | TokenKind.KwIf => true,
        TokenKind.KwWhile | TokenKind.KwFor | TokenKind.KwLoop | TokenKind.KwMatch => true,
        TokenKind.KwReturn | TokenKind.KwBreak | TokenKind.KwContinue => true,
        TokenKind.KwThrow | TokenKind.KwTry | TokenKind.KwDefer | TokenKind.KwAssert => true,
        _ => false,
    }
}

fn _starts_pattern_literal(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.Int | TokenKind.Float | TokenKind.Str | TokenKind.FStringStart => true,
        TokenKind.KwTrue | TokenKind.KwFalse | TokenKind.KwNull | TokenKind.Minus => true,
        _ => false,
    }
}

# `await` binds an operand only when one plainly follows, which is what keeps it
# a contextual keyword rather than a reserved word.
fn _starts_await_operand(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.Ident | TokenKind.KwSelf | TokenKind.Int | TokenKind.Float => true,
        TokenKind.Str | TokenKind.FStringStart => true,
        _ => false,
    }
}

# The tokens a bare Jaithon 2 `print x` argument could start with.
fn _starts_bare_argument(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.Ident | TokenKind.Int | TokenKind.Float | TokenKind.Str => true,
        TokenKind.FStringStart | TokenKind.KwTrue | TokenKind.KwFalse => true,
        TokenKind.KwNull | TokenKind.KwSelf => true,
        _ => false,
    }
}

fn _is_numeric_literal(node: Node) -> bool {
    return node.kind == NodeKind.IntLit or node.kind == NodeKind.FloatLit
}

fn _is_assign_target(node: Node) -> bool {
    return match node.kind {
        NodeKind.Ident | NodeKind.Member | NodeKind.Index | NodeKind.Slice => true,
        NodeKind.PatBind | NodeKind.PatTuple | NodeKind.PatList => true,
        NodeKind.PatWildcard => true,
        _ => false,
    }
}

fn _is_reserved_name(name: str) -> bool {
    if name.len() < 5 { return false }
    return name[0:2] == "__" and name[name.len() - 2:name.len()] == "__"
}
