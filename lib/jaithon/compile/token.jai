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
    return _KEYWORDS.get(text, TokenKind.Ident)
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

#: One token: what it is, where it is, and what it means.
pub class Token {
    #: What kind of token this is.
    pub let kind: TokenKind
    #: The byte range the token covers.
    pub let span: Span
    #: The source text the token covers, verbatim.
    pub let text: str
    #: The cooked value, or null when the kind carries no value.
    pub let value: any
    #: `TokenFlags` bits.
    pub let flags: int

    #: Record one token; only the lexer builds these.
    pub fn init(self, kind: TokenKind, span: Span, text: str, value: any = null, flags: int = 0) {
        self.kind = kind
        self.span = span
        self.text = text
        self.value = value
        self.flags = flags
    }

    #: Report whether `flag`, one of the `TokenFlags` bits, is set.
    pub fn has_flag(self, flag: int) -> bool {
        return self.flags & flag != 0
    }

    #: The cooked text of a string token; the source text of anything else.
    pub fn string_value(self) -> str {
        if isinstance(self.value, str) { return self.value }
        return self.text
    }

    fn __str__(self) -> str {
        return f"Token({kind_text(self.kind)}, {self.span.start}..{self.span.end})"
    }
}
