#: What the prompt needs from the front end that a file never asks for.
#:
#: A REPL reads one line at a time and has to decide, before compiling
#: anything, whether what it has is a whole input or the start of one. The C
#: driver used to answer that by running the C lexer and parser directly, which
#: is the last reason `src/lang` existed. Everything here replaces that: the
#: driver hands over the accumulated buffer and gets back a verdict.
#:
#: The verdict has three shapes, and telling them apart is the whole job:
#: complete (compile it), incomplete (keep reading), and mismatched (reject it
#: now, because no later line can fix a closer that opened nothing).
from jaithon.compile.lexer import Code, Diagnostic, DiagnosticBag, Lexer, Token, TokenKind
from jaithon.compile.parser import Parser

#: What is holding an input open.
#:
#: The distinction matters at the continuation prompt: a blank line abandons an
#: input that only trails an operator, but inside a bracket or a literal a
#: blank line is content, and a pasted block that contains one has to survive.
pub enum OpenKind {
    None,
    #: `(`, `[` or `{` is still waiting for its closer.
    Bracket,
    #: A triple-quoted string runs past the last line.
    String,
    #: A `#*` block comment has no `*#` yet.
    Comment,
    #: The last line ends on something that binds on.
    Operator,
}

#: The verdict on one buffer: run it, keep reading, or reject it now.
pub class Scan {
    #: A later line could still finish this input.
    pub var incomplete: bool
    #: A closer that opened nothing, which more input cannot help.
    pub var mismatched: bool
    #: What is holding it open, when `incomplete`.
    pub var open: OpenKind
    #: The bracket still open, or the one wrongly closed; empty when neither.
    pub var opener: str
    pub var opener_line: int
    #: The offending closer, when `mismatched`.
    pub var closer: str
    pub var closer_line: int
    pub var closer_col: int

    pub fn init(self) {
        self.incomplete = false
        self.mismatched = false
        self.open = OpenKind.None
        self.opener = ""
        self.opener_line = 0
        self.closer = ""
        self.closer_line = 0
        self.closer_col = 0
    }

    #: `open` as a number, for a driver that cannot name a Jaithon enum.
    #:
    #: The C prompt reads the verdict through this: the order here is the
    #: order of `JaiReplOpenKind` in `src/runtime/frontend.h`, and the two have
    #: to be changed together.
    pub fn open_ordinal(self) -> int {
        return match self.open {
            OpenKind.None => 0,
            OpenKind.Bracket => 1,
            OpenKind.String => 2,
            OpenKind.Comment => 3,
            OpenKind.Operator => 4,
        }
    }
}

fn _utf8_length(code_point: int) -> int {
    if code_point < 0x80 { return 1 }
    if code_point < 0x800 { return 2 }
    if code_point < 0x10000 { return 3 }
    return 4
}

#: The buffer indexed both ways at once.
#:
#: Every span the lexer produces is a pair of *byte* offsets, while a `str` is
#: indexed by scalar. Mixing the two silently works on ASCII and puts the caret
#: in the wrong place on anything else, so the conversion happens here and
#: nowhere else. The lexer builds the same two lists for the same reason.
class _Text {
    pub let chars: list[str]
    #: Byte offset of each character, plus the total length at the end.
    pub let offsets: list[int]

    pub fn init(self, source: str) {
        self.chars = [character for character in source]
        var byte_offset = 0
        var offsets: list[int] = []
        for character in self.chars {
            offsets.push(byte_offset)
            byte_offset += _utf8_length(ord(character))
        }
        offsets.push(byte_offset)
        self.offsets = offsets
    }

    #: Total size in bytes.
    pub fn byte_length(self) -> int { return self.offsets[self.offsets.len() - 1] }

    #: The character index a byte offset falls on, clamped to the end.
    pub fn index_of_byte(self, offset: int) -> int {
        var low = 0
        var high = self.chars.len()
        while low < high {
            let mid = (low + high) // 2
            if self.offsets[mid] < offset { low = mid + 1 } else { high = mid }
        }
        return low
    }

    # 1-based line and column of a byte offset. Columns count scalars, not
    # bytes, so a line with text above U+007F before the bracket still points
    # at the right place.
    pub fn position_at(self, offset: int) -> tuple[int, int] {
        let limit = self.index_of_byte(offset)
        var line = 1
        var column = 1
        var index = 0
        while index < limit {
            if self.chars[index] == "\n" {
                line += 1
                column = 1
            } else {
                column += 1
            }
            index += 1
        }
        return (line, column)
    }
}

fn _closer_for(opener: str) -> str {
    if opener == "(" { return ")" }
    if opener == "[" { return "]" }
    return "}"
}

fn _opener_text(kind: TokenKind) -> str? {
    if kind == TokenKind.LParen { return "(" }
    if kind == TokenKind.LBracket { return "[" }
    if kind == TokenKind.LBrace { return "{" }
    return null
}

fn _closer_text(kind: TokenKind) -> str? {
    if kind == TokenKind.RParen { return ")" }
    if kind == TokenKind.RBracket { return "]" }
    if kind == TokenKind.RBrace { return "}" }
    return null
}

#: Walk the token stream with a stack of the brackets still open. A single depth
#: counter cannot do this job: `{"a": [1}` is balanced by depth alone yet never
#: returns to zero, and clamping a stray closer at zero lets it cancel a real
#: opener later in the same input. Either way the prompt would sit waiting for a
#: line that can never arrive.
fn _scan_brackets(tokens: list[Token], text: _Text, out: Scan) -> void {
    var stack: list[tuple[str, int]] = []

    for token in tokens {
        let opener = _opener_text(token.kind)
        if opener is not null {
            stack.push((opener, token.span.start))
            continue
        }

        let closer = _closer_text(token.kind)
        if closer is null { continue }  # interpolation braces are not tokens

        if stack.len() == 0 or _closer_for(stack[stack.len() - 1][0]) != closer {
            out.mismatched = true
            out.closer = closer
            let (line, column) = text.position_at(token.span.start)
            out.closer_line = line
            out.closer_col = column
            if stack.len() > 0 {
                let top = stack[stack.len() - 1]
                out.opener = top[0]
                let (opener_line, _) = text.position_at(top[1])
                out.opener_line = opener_line
            }
            return
        }
        stack.pop()
    }

    if stack.len() > 0 {
        # Name the outermost one: that is the construct the input set out to
        # write, and the one the user is being asked to finish.
        out.incomplete = true
        out.open = OpenKind.Bracket
        out.opener = stack[0][0]
        let (opener_line, _) = text.position_at(stack[0][1])
        out.opener_line = opener_line
    }
}

#: True when the quote at byte `offset`, past any `r`/`f` prefix, is tripled.
fn _is_triple_quote_at(text: _Text, offset: int) -> bool {
    var index = text.index_of_byte(offset)
    let length = text.chars.len()
    while index < length and (text.chars[index] == "r" or text.chars[index] == "f") {
        index += 1
    }
    if index + 2 >= length { return false }
    let quote = text.chars[index]
    if quote != "\"" and quote != "'" { return false }
    return text.chars[index + 1] == quote and text.chars[index + 2] == quote
}

#: Where the f-string the lexer is still inside was opened, or -1.
#:
#: An f-string diagnostic is raised where the lexer noticed the trouble, not at
#: the quote that started the literal: `f"""a {n}` reports at the byte after the
#: interpolation, and an interpolation still open at end of input reports at the
#: end of the buffer. Neither offset says whether the literal was triple-quoted,
#: so the opener has to come from the token stream, where every f-string leaves
#: an `FStringStart` at its first byte. The last one is the one still open: the
#: lexer only stops mid-literal at the end of the input.
fn _open_fstring_offset(tokens: list[Token]) -> int {
    var offset = -1
    for token in tokens {
        if token.kind == TokenKind.FStringStart { offset = token.span.start }
    }
    return offset
}

#: An unterminated literal only means "keep typing" when a later line could
#: still finish it: a block comment, or a triple-quoted string still open where
#: the input stops. The decision is made on the span of the literal the lexer
#: complained about rather than on the whole buffer, so a complete `"""..."""`
#: earlier in the input cannot make a broken one-line string look like the start
#: of a long one.
fn _unterminated_literal_continues(
    bag: DiagnosticBag,
    tokens: list[Token],
    text: _Text,
    out: Scan
) -> bool {
    if bag.items.len() == 0 { return false }

    var last: Diagnostic? = null
    for diagnostic in bag.items {
        if diagnostic.code == Code.E0002_UNTERMINATED_COMMENT { continue }
        if diagnostic.code != Code.E0001_UNTERMINATED_STRING and
            diagnostic.code != Code.E0008_UNTERMINATED_INTERPOLATION {
            return false
        }
        last = diagnostic
    }

    if last is null {  # block comments only
        out.open = OpenKind.Comment
        out.opener = "#"
        let first = bag.items[0].span
        let start = if first is null { 0 } else { first.start }
        let (line, _) = text.position_at(start)
        out.opener_line = line
        return true
    }

    let span = last.span
    if span is null { return false }
    var opener = span.start
    if last.code == Code.E0008_UNTERMINATED_INTERPOLATION {
        let found = _open_fstring_offset(tokens)
        if found < 0 { return false }
        opener = found
    } else {
        # The literal has to reach the end of the input; one the lexer recovered
        # from at an earlier newline is a plain error. The buffer always ends in
        # the newline the prompt appended, which the span stops short of. An
        # f-string needs no such test: the lexer leaves a triple-quoted one open
        # only when it runs out of input.
        if span.end + 1 < text.byte_length() { return false }
    }
    if not _is_triple_quote_at(text, opener) { return false }

    out.open = OpenKind.String
    out.opener = "\""
    let (line, _) = text.position_at(opener)
    out.opener_line = line
    return true
}

#: Decide what to do with `source`: run it, keep reading, or reject it now.
#:
#: Nothing is registered and no diagnostic escapes: a line still being typed
#: must not add to the source table, and the compile that follows a complete
#: input produces the real diagnostics with real spans.
#:
#: ```
#: scan("1 + 1\n").incomplete       # false
#: scan("f(1,\n").incomplete        # true
#: scan("1)\n").mismatched          # true
#: ```
pub fn scan(source: str) -> Scan {
    let out = Scan()
    let text = _Text(source)
    let bag = DiagnosticBag()
    let lexer = Lexer(source, -1, bag)
    let tokens = lexer.tokenize()

    if not lexer.had_error {
        _scan_brackets(tokens, text, out)
        if not out.incomplete and not out.mismatched {
            # Brackets balance, so the only thing left that can hold the input
            # open is a line ending on a token the next one continues.
            let parser = Parser(Lexer(source, -1, DiagnosticBag()))
            let (_, incomplete) = parser.parse_repl_line()
            if incomplete {
                out.incomplete = true
                out.open = OpenKind.Operator
            }
        }
    } elif _unterminated_literal_continues(bag, tokens, text, out) {
        out.incomplete = true
    }

    return out
}
