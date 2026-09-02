#: The tokens the scanner produces, one class for all twenty kinds.
#:
#: PyYAML has a class per token; here one `Token` carries a `TokenKind` and
#: the fields the kinds share, which keeps the scanner's `check_token` a
#: comparison of enum values rather than a walk over classes. `id` returns
#: PyYAML's name for the kind, which is what its error messages quote.
from .error import Mark

#: Which token this is.
pub enum TokenKind {
    Directive,
    DocumentStart,
    DocumentEnd,
    StreamStart,
    StreamEnd,
    BlockSequenceStart,
    BlockMappingStart,
    BlockEnd,
    FlowSequenceStart,
    FlowMappingStart,
    FlowSequenceEnd,
    FlowMappingEnd,
    Key,
    Value,
    BlockEntry,
    FlowEntry,
    Alias,
    Anchor,
    Tag,
    Scalar,
}

#: PyYAML's `id` string for a kind, as quoted in parser errors.
pub fn token_id(kind: TokenKind) -> str {
    return match kind {
        TokenKind.Directive => "<directive>",
        TokenKind.DocumentStart => "<document start>",
        TokenKind.DocumentEnd => "<document end>",
        TokenKind.StreamStart => "<stream start>",
        TokenKind.StreamEnd => "<stream end>",
        TokenKind.BlockSequenceStart => "<block sequence start>",
        TokenKind.BlockMappingStart => "<block mapping start>",
        TokenKind.BlockEnd => "<block end>",
        TokenKind.FlowSequenceStart => "[",
        TokenKind.FlowMappingStart => "{",
        TokenKind.FlowSequenceEnd => "]",
        TokenKind.FlowMappingEnd => "}",
        TokenKind.Key => "?",
        TokenKind.Value => ":",
        TokenKind.BlockEntry => "-",
        TokenKind.FlowEntry => ",",
        TokenKind.Alias => "<alias>",
        TokenKind.Anchor => "<anchor>",
        TokenKind.Tag => "<tag>",
        TokenKind.Scalar => "<scalar>",
    }
}

#: One token and where it sits.
#:
#: `value` is the alias or anchor name, the scalar text, or a directive's
#: name. A scalar also has `plain` and `style`; a tag has `handle` and
#: `suffix`; a `%YAML` directive has `major` and `minor`, and a `%TAG`
#: directive reuses `handle` and `suffix` for its handle and prefix.
pub class Token {
    #: Which kind this is.
    pub let kind: TokenKind

    #: Where the token starts.
    pub let start_mark: Mark

    #: Where the token ends.
    pub let end_mark: Mark

    #: Scalar text, alias or anchor name, or directive name.
    pub var value: str

    #: Whether a scalar was written plain, without quotes or a block header.
    pub var plain: bool

    #: A scalar's style: `'`, `"`, `|` or `>`, and null when plain.
    pub var style: str?

    #: A tag's handle, or a `%TAG` directive's handle; null for a verbatim tag.
    pub var handle: str?

    #: A tag's suffix, or a `%TAG` directive's prefix.
    pub var suffix: str?

    #: A `%YAML` directive's major version.
    pub var major: int

    #: A `%YAML` directive's minor version.
    pub var minor: int

    #: A token of `kind` spanning the two marks.
    pub fn init(self, kind: TokenKind, start_mark: Mark, end_mark: Mark) {
        self.kind = kind
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.value = ""
        self.plain = false
        self.style = null
        self.handle = null
        self.suffix = null
        self.major = 0
        self.minor = 0
    }

    #: PyYAML's name for the kind.
    pub fn id(self) -> str { return token_id(self.kind) }

    fn __str__(self) -> str { return f"Token({self.kind}, {repr(self.value)})" }
}
