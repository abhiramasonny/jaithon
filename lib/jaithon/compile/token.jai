#: What a token is: its `TokenKind` vocabulary, the `TokenFlags` bits it may
#: carry, the `Token` record itself, and the pure lookups over a kind (the
#: keyword table, the C name, the display spelling).
#:
#: Split out of `lexer.jai`: none of this touches lexer state (`position`,
#: current character, line tracking) — it describes tokens, not how source
#: text becomes them. Scanning stays in `lexer.jai`; this file is its output
#: alphabet.
from jaithon.ast import Span

#: Bit flags carried by `Token.flags`, mirroring `TOKF_*` in `src/lang/token.h`.
pub class TokenFlags {
    #: The token is the first one on its line.
    pub static let AFTER_NEWLINE: int = 1
    #: The literal contained a backslash escape that was cooked away.
    pub static let HAS_ESCAPE: int = 2
    #: The literal was written `r"..."`.
    pub static let RAW_STRING: int = 4
    #: The literal was written with three quotes.
    pub static let TRIPLE_STRING: int = 8
    #: An `Int` holding 9223372036854775808, the magnitude of the smallest
    #: `int`. It is only accepted directly behind a prefix `-` and its value is
    #: already negated, so the parser swallows that `-` instead of emitting a
    #: negation that could not be performed.
    pub static let NEG_MAGNITUDE: int = 16
}

#: Every token kind, in the order of `src/lang/token.h`.
pub enum TokenKind {
    Eof,
    Error,
    Newline,

    Int,
    Float,
    Str,
    FStringStart,
    FStringMid,
    FStringEnd,
    Ident,
    Label,

    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Dot,
    DotDot,
    DotDotEq,
    Ellipsis,
    Colon,
    Semicolon,
    Arrow,
    FatArrow,
    At,
    Question,
    QuestionDot,
    QuestionQuestion,
    Underscore,

    Plus,
    Minus,
    Star,
    Slash,
    SlashSlash,
    Percent,
    StarStar,
    PlusPercent,
    MinusPercent,
    StarPercent,
    Amp,
    Pipe,
    Caret,
    Tilde,
    Shl,
    Shr,
    Eq,
    EqEq,
    BangEq,
    Lt,
    Le,
    Gt,
    Ge,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    SlashSlashEq,
    PercentEq,
    StarStarEq,
    AmpEq,
    PipeEq,
    CaretEq,
    ShlEq,
    ShrEq,

    KwAnd,
    KwAs,
    KwAssert,
    KwBreak,
    KwCase,
    KwCatch,
    KwClass,
    KwConst,
    KwContinue,
    KwDefer,
    KwElif,
    KwElse,
    KwEnum,
    KwExport,
    KwExtends,
    KwFalse,
    KwFinally,
    KwFor,
    KwFn,
    KwFrom,
    KwIf,
    KwImpl,
    KwImport,
    KwIn,
    KwIs,
    KwLet,
    KwLoop,
    KwMatch,
    KwModule,
    KwMut,
    KwNot,
    KwNull,
    KwOr,
    KwProt,
    KwPub,
    KwReturn,
    KwSelf,
    KwStatic,
    KwSuper,
    KwThrow,
    KwTrait,
    KwTrue,
    KwTry,
    KwType,
    KwVar,
    KwWhile,
    KwYield,
}

let _KEYWORDS: dict[str, TokenKind] = {
    "and": TokenKind.KwAnd,
    "as": TokenKind.KwAs,
    "assert": TokenKind.KwAssert,
    "break": TokenKind.KwBreak,
    "case": TokenKind.KwCase,
    "catch": TokenKind.KwCatch,
    "class": TokenKind.KwClass,
    "const": TokenKind.KwConst,
    "continue": TokenKind.KwContinue,
    "defer": TokenKind.KwDefer,
    "elif": TokenKind.KwElif,
    "else": TokenKind.KwElse,
    "enum": TokenKind.KwEnum,
    "export": TokenKind.KwExport,
    "extends": TokenKind.KwExtends,
    "false": TokenKind.KwFalse,
    "finally": TokenKind.KwFinally,
    "for": TokenKind.KwFor,
    "fn": TokenKind.KwFn,
    "from": TokenKind.KwFrom,
    "if": TokenKind.KwIf,
    "impl": TokenKind.KwImpl,
    "import": TokenKind.KwImport,
    "in": TokenKind.KwIn,
    "is": TokenKind.KwIs,
    "let": TokenKind.KwLet,
    "loop": TokenKind.KwLoop,
    "match": TokenKind.KwMatch,
    "module": TokenKind.KwModule,
    "mut": TokenKind.KwMut,
    "not": TokenKind.KwNot,
    "null": TokenKind.KwNull,
    "or": TokenKind.KwOr,
    "prot": TokenKind.KwProt,
    "pub": TokenKind.KwPub,
    "return": TokenKind.KwReturn,
    "self": TokenKind.KwSelf,
    "static": TokenKind.KwStatic,
    "super": TokenKind.KwSuper,
    "throw": TokenKind.KwThrow,
    "trait": TokenKind.KwTrait,
    "true": TokenKind.KwTrue,
    "try": TokenKind.KwTry,
    "type": TokenKind.KwType,
    "var": TokenKind.KwVar,
    "while": TokenKind.KwWhile,
    "yield": TokenKind.KwYield,
}

#: Return the keyword kind spelled `text`, or `TokenKind.Ident`.
#:
#: ```
#: keyword_lookup("fn") == TokenKind.KwFn      # true
#: keyword_lookup("fns") == TokenKind.Ident    # true
#: ```
pub fn keyword_lookup(text: str) -> TokenKind {
    # A test and an index rather than `.get(text, default)`: `get` answers a
    # value the checker cannot type, so the return needed a `TokenKind` guard
    # the tier refuses -- and a call to an uncompiled function cuts the
    # compiled walk of its caller short, which was the lexer's identifier scan.
    if text in _KEYWORDS { return _KEYWORDS[text] }
    return TokenKind.Ident
}
#: The C name of each kind, as `jaiTokenKindName` in `src/lang/token.c` spells
#: it. `jaithon tokens` prints these, and the spelling is not derivable from the
#: variant: `Str` is `TOK_STRING`, not `TOK_STR`.
let _KIND_NAMES: dict[TokenKind, str] = {
    TokenKind.Eof: "TOK_EOF",
    TokenKind.Error: "TOK_ERROR",
    TokenKind.Newline: "TOK_NEWLINE",
    TokenKind.Int: "TOK_INT",
    TokenKind.Float: "TOK_FLOAT",
    TokenKind.Str: "TOK_STRING",
    TokenKind.FStringStart: "TOK_FSTRING_START",
    TokenKind.FStringMid: "TOK_FSTRING_MID",
    TokenKind.FStringEnd: "TOK_FSTRING_END",
    TokenKind.Ident: "TOK_IDENT",
    TokenKind.Label: "TOK_LABEL",
    TokenKind.LParen: "TOK_LPAREN",
    TokenKind.RParen: "TOK_RPAREN",
    TokenKind.LBracket: "TOK_LBRACKET",
    TokenKind.RBracket: "TOK_RBRACKET",
    TokenKind.LBrace: "TOK_LBRACE",
    TokenKind.RBrace: "TOK_RBRACE",
    TokenKind.Comma: "TOK_COMMA",
    TokenKind.Dot: "TOK_DOT",
    TokenKind.DotDot: "TOK_DOTDOT",
    TokenKind.DotDotEq: "TOK_DOTDOT_EQ",
    TokenKind.Ellipsis: "TOK_ELLIPSIS",
    TokenKind.Colon: "TOK_COLON",
    TokenKind.Semicolon: "TOK_SEMICOLON",
    TokenKind.Arrow: "TOK_ARROW",
    TokenKind.FatArrow: "TOK_FAT_ARROW",
    TokenKind.At: "TOK_AT",
    TokenKind.Question: "TOK_QUESTION",
    TokenKind.QuestionDot: "TOK_QUESTION_DOT",
    TokenKind.QuestionQuestion: "TOK_QUESTION_QUESTION",
    TokenKind.Underscore: "TOK_UNDERSCORE",
    TokenKind.Plus: "TOK_PLUS",
    TokenKind.Minus: "TOK_MINUS",
    TokenKind.Star: "TOK_STAR",
    TokenKind.Slash: "TOK_SLASH",
    TokenKind.SlashSlash: "TOK_SLASH_SLASH",
    TokenKind.Percent: "TOK_PERCENT",
    TokenKind.StarStar: "TOK_STAR_STAR",
    TokenKind.PlusPercent: "TOK_PLUS_PERCENT",
    TokenKind.MinusPercent: "TOK_MINUS_PERCENT",
    TokenKind.StarPercent: "TOK_STAR_PERCENT",
    TokenKind.Amp: "TOK_AMP",
    TokenKind.Pipe: "TOK_PIPE",
    TokenKind.Caret: "TOK_CARET",
    TokenKind.Tilde: "TOK_TILDE",
    TokenKind.Shl: "TOK_SHL",
    TokenKind.Shr: "TOK_SHR",
    TokenKind.Eq: "TOK_EQ",
    TokenKind.EqEq: "TOK_EQ_EQ",
    TokenKind.BangEq: "TOK_BANG_EQ",
    TokenKind.Lt: "TOK_LT",
    TokenKind.Le: "TOK_LE",
    TokenKind.Gt: "TOK_GT",
    TokenKind.Ge: "TOK_GE",
    TokenKind.PlusEq: "TOK_PLUS_EQ",
    TokenKind.MinusEq: "TOK_MINUS_EQ",
    TokenKind.StarEq: "TOK_STAR_EQ",
    TokenKind.SlashEq: "TOK_SLASH_EQ",
    TokenKind.SlashSlashEq: "TOK_SLASH_SLASH_EQ",
    TokenKind.PercentEq: "TOK_PERCENT_EQ",
    TokenKind.StarStarEq: "TOK_STAR_STAR_EQ",
    TokenKind.AmpEq: "TOK_AMP_EQ",
    TokenKind.PipeEq: "TOK_PIPE_EQ",
    TokenKind.CaretEq: "TOK_CARET_EQ",
    TokenKind.ShlEq: "TOK_SHL_EQ",
    TokenKind.ShrEq: "TOK_SHR_EQ",
    TokenKind.KwAnd: "TOK_KW_AND",
    TokenKind.KwAs: "TOK_KW_AS",
    TokenKind.KwAssert: "TOK_KW_ASSERT",
    TokenKind.KwBreak: "TOK_KW_BREAK",
    TokenKind.KwCase: "TOK_KW_CASE",
    TokenKind.KwCatch: "TOK_KW_CATCH",
    TokenKind.KwClass: "TOK_KW_CLASS",
    TokenKind.KwConst: "TOK_KW_CONST",
    TokenKind.KwContinue: "TOK_KW_CONTINUE",
    TokenKind.KwDefer: "TOK_KW_DEFER",
    TokenKind.KwElif: "TOK_KW_ELIF",
    TokenKind.KwElse: "TOK_KW_ELSE",
    TokenKind.KwEnum: "TOK_KW_ENUM",
    TokenKind.KwExport: "TOK_KW_EXPORT",
    TokenKind.KwExtends: "TOK_KW_EXTENDS",
    TokenKind.KwFalse: "TOK_KW_FALSE",
    TokenKind.KwFinally: "TOK_KW_FINALLY",
    TokenKind.KwFor: "TOK_KW_FOR",
    TokenKind.KwFn: "TOK_KW_FN",
    TokenKind.KwFrom: "TOK_KW_FROM",
    TokenKind.KwIf: "TOK_KW_IF",
    TokenKind.KwImpl: "TOK_KW_IMPL",
    TokenKind.KwImport: "TOK_KW_IMPORT",
    TokenKind.KwIn: "TOK_KW_IN",
    TokenKind.KwIs: "TOK_KW_IS",
    TokenKind.KwLet: "TOK_KW_LET",
    TokenKind.KwLoop: "TOK_KW_LOOP",
    TokenKind.KwMatch: "TOK_KW_MATCH",
    TokenKind.KwModule: "TOK_KW_MODULE",
    TokenKind.KwMut: "TOK_KW_MUT",
    TokenKind.KwNot: "TOK_KW_NOT",
    TokenKind.KwNull: "TOK_KW_NULL",
    TokenKind.KwOr: "TOK_KW_OR",
    TokenKind.KwProt: "TOK_KW_PROT",
    TokenKind.KwPub: "TOK_KW_PUB",
    TokenKind.KwReturn: "TOK_KW_RETURN",
    TokenKind.KwSelf: "TOK_KW_SELF",
    TokenKind.KwStatic: "TOK_KW_STATIC",
    TokenKind.KwSuper: "TOK_KW_SUPER",
    TokenKind.KwThrow: "TOK_KW_THROW",
    TokenKind.KwTrait: "TOK_KW_TRAIT",
    TokenKind.KwTrue: "TOK_KW_TRUE",
    TokenKind.KwTry: "TOK_KW_TRY",
    TokenKind.KwType: "TOK_KW_TYPE",
    TokenKind.KwVar: "TOK_KW_VAR",
    TokenKind.KwWhile: "TOK_KW_WHILE",
    TokenKind.KwYield: "TOK_KW_YIELD",
}

#: The name `jaithon tokens` prints for `kind`.
#:
#: ```
#: kind_name(TokenKind.KwLet)     # "TOK_KW_LET"
#: ```
pub fn kind_name(kind: TokenKind) -> str {
    return _KIND_NAMES.get(kind, "TOK_INVALID")
}

#: Return how `kind` is written in source, or a placeholder for the kinds whose
#: text varies. These strings land verbatim in diagnostics.
#:
#: ```
#: kind_text(TokenKind.StarStar)    # "**"
#: kind_text(TokenKind.Ident)       # "<identifier>"
#: ```
pub fn kind_text(kind: TokenKind) -> str {
    return match kind {
        TokenKind.Eof => "<end of file>",
        TokenKind.Error => "<error>",
        TokenKind.Newline => "<newline>",
        TokenKind.Int => "<integer literal>",
        TokenKind.Float => "<float literal>",
        TokenKind.Str => "<string literal>",
        TokenKind.FStringStart => "<f-string>",
        TokenKind.FStringMid => "<f-string part>",
        TokenKind.FStringEnd => "<f-string end>",
        TokenKind.Ident => "<identifier>",
        TokenKind.Label => "<label>",
        TokenKind.LParen => "(",
        TokenKind.RParen => ")",
        TokenKind.LBracket => "[",
        TokenKind.RBracket => "]",
        TokenKind.LBrace => "{",
        TokenKind.RBrace => "}",
        TokenKind.Comma => ",",
        TokenKind.Dot => ".",
        TokenKind.DotDot => "..",
        TokenKind.DotDotEq => "..=",
        TokenKind.Ellipsis => "...",
        TokenKind.Colon => ":",
        TokenKind.Semicolon => ";",
        TokenKind.Arrow => "->",
        TokenKind.FatArrow => "=>",
        TokenKind.At => "@",
        TokenKind.Question => "?",
        TokenKind.QuestionDot => "?.",
        TokenKind.QuestionQuestion => "??",
        TokenKind.Underscore => "_",
        TokenKind.Plus => "+",
        TokenKind.Minus => "-",
        TokenKind.Star => "*",
        TokenKind.Slash => "/",
        TokenKind.SlashSlash => "//",
        TokenKind.Percent => "%",
        TokenKind.StarStar => "**",
        TokenKind.PlusPercent => "+%",
        TokenKind.MinusPercent => "-%",
        TokenKind.StarPercent => "*%",
        TokenKind.Amp => "&",
        TokenKind.Pipe => "|",
        TokenKind.Caret => "^",
        TokenKind.Tilde => "~",
        TokenKind.Shl => "<<",
        TokenKind.Shr => ">>",
        TokenKind.Eq => "=",
        TokenKind.EqEq => "==",
        TokenKind.BangEq => "!=",
        TokenKind.Lt => "<",
        TokenKind.Le => "<=",
        TokenKind.Gt => ">",
        TokenKind.Ge => ">=",
        TokenKind.PlusEq => "+=",
        TokenKind.MinusEq => "-=",
        TokenKind.StarEq => "*=",
        TokenKind.SlashEq => "/=",
        TokenKind.SlashSlashEq => "//=",
        TokenKind.PercentEq => "%=",
        TokenKind.StarStarEq => "**=",
        TokenKind.AmpEq => "&=",
        TokenKind.PipeEq => "|=",
        TokenKind.CaretEq => "^=",
        TokenKind.ShlEq => "<<=",
        TokenKind.ShrEq => ">>=",
        TokenKind.KwAnd => "and",
        TokenKind.KwAs => "as",
        TokenKind.KwAssert => "assert",
        TokenKind.KwBreak => "break",
        TokenKind.KwCase => "case",
        TokenKind.KwCatch => "catch",
        TokenKind.KwClass => "class",
        TokenKind.KwConst => "const",
        TokenKind.KwContinue => "continue",
        TokenKind.KwDefer => "defer",
        TokenKind.KwElif => "elif",
        TokenKind.KwElse => "else",
        TokenKind.KwEnum => "enum",
        TokenKind.KwExport => "export",
        TokenKind.KwExtends => "extends",
        TokenKind.KwFalse => "false",
        TokenKind.KwFinally => "finally",
        TokenKind.KwFor => "for",
        TokenKind.KwFn => "fn",
        TokenKind.KwFrom => "from",
        TokenKind.KwIf => "if",
        TokenKind.KwImpl => "impl",
        TokenKind.KwImport => "import",
        TokenKind.KwIn => "in",
        TokenKind.KwIs => "is",
        TokenKind.KwLet => "let",
        TokenKind.KwLoop => "loop",
        TokenKind.KwMatch => "match",
        TokenKind.KwModule => "module",
        TokenKind.KwMut => "mut",
        TokenKind.KwNot => "not",
        TokenKind.KwNull => "null",
        TokenKind.KwOr => "or",
        TokenKind.KwProt => "prot",
        TokenKind.KwPub => "pub",
        TokenKind.KwReturn => "return",
        TokenKind.KwSelf => "self",
        TokenKind.KwStatic => "static",
        TokenKind.KwSuper => "super",
        TokenKind.KwThrow => "throw",
        TokenKind.KwTrait => "trait",
        TokenKind.KwTrue => "true",
        TokenKind.KwTry => "try",
        TokenKind.KwType => "type",
        TokenKind.KwVar => "var",
        TokenKind.KwWhile => "while",
        TokenKind.KwYield => "yield",
    }
}

#: The file a token was cut from, shared by every token one `Lexer` makes.
#:
#: A token points at one of these instead of carrying a `Span` and a
#: substring of its own, so neither is built until a reader asks for it.
#:
#: The two coordinate systems disagree: a `Span` records UTF-8 byte offsets,
#: while a `str` is indexed by character. When the file is all ASCII they
#: coincide and `slice` indexes `text` directly; otherwise `scalar_of` -- the
#: lexer's own table, not a copy -- maps a byte offset to its character.
pub class Source {
    #: The file id every `Span` cut from this source carries, or -1.
    pub let file: int
    #: The source text, exactly as the lexer received it.
    pub let text: str
    let ascii: bool
    #: When `text` is not ASCII: the character index of each byte offset, and
    #: of the end. Otherwise `[0]` and never read.
    let scalar_of: list[int]

    #: Describe the file numbered `file`, whose text is `text`.
    pub fn init(self, file: int, text: str, ascii: bool, scalar_of: list[int]) {
        self.file = file
        self.text = text
        self.ascii = ascii
        self.scalar_of = scalar_of
    }

    #: The source text between two byte offsets.
    pub fn slice(self, start: int, end: int) -> str {
        if self.ascii { return self.text[start:end] }
        return self.text[self.scalar_of[start]:self.scalar_of[end]]
    }
}

#: One token: what it is, where it is, and what it means.
#:
#: A token records the byte range it covers and the `Source` it was cut from,
#: not a `Span` and a substring: most tokens -- punctuation, most keywords, the
#: newlines -- are read for their kind and nothing else, and paid for both.
#: `span` and `text` are computed when read, so a reader spells them as it
#: always did.
#:
#: A getter read allocates a bound method and calls back into the interpreter,
#: so a reader on a hot path does not use one: it reads `word`, `start` and
#: `end`, or calls `make_span` and `slice_text`.
pub class Token {
    #: What kind of token this is.
    pub let kind: TokenKind
    #: The file the token was cut from.
    pub let source: Source
    #: Byte offset of the token's first byte.
    pub let start: int
    #: Byte offset one past the token's last byte.
    pub let end: int
    #: The text of a word -- an identifier, a keyword or `_` -- and "" for
    #: every other kind.
    #:
    #: The one substring a token still holds, because it costs nothing: the
    #: lexer has already cut it to look the word up as a keyword, and the
    #: parser reads nearly every identifier's name. Cutting it again from
    #: `source` would be a second slice per name, and a call the compiled tier
    #: cannot make, since `source` is an object it reads out of a field.
    pub let word: str
    #: The cooked value, or null when the kind carries no value. The lexer
    #: sets it after construction, on the literal kinds that carry one.
    pub var value: any
    #: `TokenFlags` bits.
    pub let flags: int

    #: Record one token; only the lexer builds these.
    #:
    #: The value is not a parameter. A parameter that is an int on one call,
    #: a string on the next and null on most has no kind the tier can give it,
    #: and that one parameter kept every token's construction interpreted.
    pub fn init(
        self,
        kind: TokenKind,
        source: Source,
        start: int,
        end: int,
        word: str = "",
        flags: int = 0
    ) {
        self.kind = kind
        self.source = source
        self.start = start
        self.end = end
        self.word = word
        self.value = null
        self.flags = flags
    }

    #: The byte range the token covers.
    pub get span() -> Span {
        return self.make_span()
    }

    #: The source text the token covers, verbatim.
    pub get text() -> str {
        return self.slice_text()
    }

    #: `span`, built now: a new `Span` on every call.
    pub fn make_span(self) -> Span {
        return Span(self.source.file, self.start, self.end)
    }

    #: `text`, without the getter: the word, or a slice of the source.
    pub fn slice_text(self) -> str {
        if self.word.len() > 0 { return self.word }
        return self.source.slice(self.start, self.end)
    }

    #: Report whether `flag`, one of the `TokenFlags` bits, is set.
    pub fn has_flag(self, flag: int) -> bool {
        return self.flags & flag != 0
    }

    #: The cooked text of a string token; the source text of anything else.
    pub fn string_value(self) -> str {
        if isinstance(self.value, str) { return self.value }
        return self.slice_text()
    }

    fn __str__(self) -> str {
        return f"Token({kind_text(self.kind)}, {self.start}..{self.end})"
    }
}
