#: Diagnostics shared by the whole front end: a severity, one `Diagnostic`,
#: the `DiagnosticBag` that collects them, and the stable `Code` registry.
#:
#: Split out of `lexer.jai`: none of this carries lexer (or parser) state —
#: it's the vocabulary both halves of the front end report through. `Code` in
#: particular is raised by the lexer (`E00xx`), the parser (`E01xx`), and
#: later stages alike (`E0304`, `E0804`, `W0105`), so it lives here rather
#: than under either producer's file, and each half names an error exactly
#: once.
from jaithon.ast import Span

#: Stable diagnostic codes, spelled as in `src/common/diag.h` (spec §13).
pub class Code {
    pub static let E0001_UNTERMINATED_STRING: str = "E0001"
    pub static let E0002_UNTERMINATED_COMMENT: str = "E0002"
    pub static let E0003_INVALID_CHARACTER: str = "E0003"
    pub static let E0004_INVALID_ESCAPE: str = "E0004"
    pub static let E0005_INVALID_NUMBER: str = "E0005"
    pub static let E0008_UNTERMINATED_INTERPOLATION: str = "E0008"
    pub static let E0009_NESTED_INTERPOLATION_TOO_DEEP: str = "E0009"
    pub static let E0100_UNEXPECTED_TOKEN: str = "E0100"
    pub static let E0101_EXPECTED_TOKEN: str = "E0101"
    pub static let E0102_EXPECTED_EXPRESSION: str = "E0102"
    pub static let E0103_EXPECTED_STATEMENT: str = "E0103"
    pub static let E0104_EXPECTED_BLOCK: str = "E0104"
    pub static let E0105_EXPECTED_TYPE: str = "E0105"
    pub static let E0106_EXPECTED_IDENTIFIER: str = "E0106"
    pub static let E0107_UNCLOSED_DELIMITER: str = "E0107"
    pub static let E0108_TRAILING_TOKENS: str = "E0108"
    pub static let E0109_INVALID_ASSIGN_TARGET: str = "E0109"
    pub static let E0110_INVALID_PATTERN: str = "E0110"
    pub static let E0111_DUPLICATE_PARAMETER: str = "E0111"
    pub static let E0112_PARAM_AFTER_VARIADIC: str = "E0112"
    pub static let E0113_DEFAULT_BEFORE_REQUIRED: str = "E0113"
    pub static let E0114_RESERVED_IDENTIFIER: str = "E0114"
    pub static let E0115_LABEL_NOT_ON_LOOP: str = "E0115"
    pub static let E0304_MISSING_INITIALIZER: str = "E0304"
    pub static let E0804_INVALID_MODULE_PATH: str = "E0804"
    pub static let W0105_DEPRECATED: str = "W0105"
}

#: How loudly a diagnostic speaks.
pub enum Severity { Error, Warning, Note, Help }

#: One diagnostic: a stable code, a message, and the span it underlines.
#:
#: `label`, `with_help` and `with_note` return the diagnostic, so the extra
#: detail reads as one expression at the site that raised it.
pub class Diagnostic {
    #: The stable code, for example `"E0005"`.
    pub let code: str
    #: Error or warning.
    pub let severity: Severity
    #: The one-line message, lowercase and without a trailing period.
    pub let message: str
    #: The span the caret underlines, or null for a whole-file diagnostic.
    pub let span: Span?
    #: Secondary underlines, as `(span, label)` pairs; the label may be null.
    pub var labels: list[tuple[Span, str?]]
    #: The `help:` line, when there is a concrete fix to suggest.
    pub var help: str?
    #: The `note:` line, when background is needed.
    pub var note: str?

    #: Record one diagnostic; `related` becomes an unlabelled secondary span.
    pub fn init(
        self,
        code: str,
        severity: Severity,
        message: str,
        span: Span?,
        help: str? = null,
        related: Span? = null
    ) {
        self.code = code
        self.severity = severity
        self.message = message
        self.span = span
        self.labels = []
        self.help = help
        self.note = null
        if related != null { self.labels.push((related, null)) }
    }

    #: Underline `span` as well, with `text` beside it.
    pub fn label(self, span: Span, text: str?) -> Diagnostic {
        self.labels.push((span, text))
        return self
    }

    #: Attach the `help:` line.
    pub fn with_help(self, text: str) -> Diagnostic {
        self.help = text
        return self
    }

    #: Attach the `note:` line.
    pub fn with_note(self, text: str) -> Diagnostic {
        self.note = text
        return self
    }

    #: Report whether this diagnostic stops compilation.
    pub fn is_error(self) -> bool {
        return self.severity == Severity.Error
    }

    fn __str__(self) -> str {
        let label = match self.severity {
            Severity.Error => "error",
            Severity.Warning => "warning",
            Severity.Note => "note",
            Severity.Help => "help",
        }
        return f"{label}[{self.code}]: {self.message}"
    }
}

#: Everything reported while compiling one file.
#:
#: The lexer and the parser share one bag, so a file's diagnostics come out in
#: source order regardless of which half produced them.
pub class DiagnosticBag {
    #: The diagnostics recorded so far, in the order they were raised.
    pub var items: list[Diagnostic]
    #: How many errors were raised, including any past `max_errors`.
    pub var error_count: int
    #: How many warnings were raised.
    pub var warning_count: int
    #: Stop recording after this many errors; counting continues.
    pub var max_errors: int

    #: Create an empty bag that stops recording after `max_errors` errors.
    pub fn init(self, max_errors: int = 25) {
        if max_errors < 1 { throw ValueError(f"max_errors must be positive, got {max_errors}") }
        self.items = []
        self.error_count = 0
        self.warning_count = 0
        self.max_errors = max_errors
    }

    #: Record an error and return it so the caller can attach labels.
    pub fn error(
        self,
        code: str,
        span: Span?,
        message: str,
        help: str? = null,
        note: str? = null
    ) -> Diagnostic {
        let diag = Diagnostic(code, Severity.Error, message, span, help)
        diag.note = note
        self.error_count += 1
        if self.items.len() < self.max_errors { self.items.push(diag) }
        return diag
    }

    #: Record a warning and return it.
    pub fn warn(
        self,
        code: str,
        span: Span?,
        message: str,
        help: str? = null,
        note: str? = null
    ) -> Diagnostic {
        let diag = Diagnostic(code, Severity.Warning, message, span, help)
        diag.note = note
        self.warning_count += 1
        self.items.push(diag)
        return diag
    }

    #: Report whether any error was raised.
    pub fn has_errors(self) -> bool {
        return self.error_count > 0
    }
}
