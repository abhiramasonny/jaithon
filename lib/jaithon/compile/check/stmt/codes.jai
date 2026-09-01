#: Everything a statement can report, and the caps it reports within.
#:
#: The bottom of `stmt/`: it imports nothing, so any file of the directory may
#: name it. The codes are gathered here rather than declared beside the check
#: that raises them because they are the surface of the whole directory —
#: `stmt.jai` re-exports this list unchanged, and a reader comparing it against
#: `src/common/diag.h` wants one file rather than five.
#:
#: What is NOT here: the codes `expr.jai` and `ctx.jai` own — `E0400`, `E0404`,
#: `E0406`, `E0410`, `E0411`, `E0701` and `W0102`. Statements raise all of them
#: and import each from the module that declared it.

#: A store into something declared `let`.
#:
#: The codes below are the `JaiDiagCode` names in `src/common/diag.h`. Both the
#: code and the message text are a contract: Phase 4 turns `tests/errors` into
#: goldens and a reworded message fails there.
pub const E0301_ASSIGN_TO_IMMUTABLE: str = "E0301"

#: A `const` whose initialiser has no compile-time value.
pub const E0303_NON_CONSTANT_CONST: str = "E0303"

#: A `const` with no initialiser at all.
pub const E0304_MISSING_INITIALIZER: str = "E0304"

#: A pattern naming a type this unit cannot find.
pub const E0402_UNKNOWN_TYPE: str = "E0402"

#: A `return` that does not fit the enclosing function's declared result.
pub const E0408_RETURN_TYPE_MISMATCH: str = "E0408"

#: A match arm an earlier arm already takes.
pub const E0500_UNREACHABLE_ARM: str = "E0500"

#: A `match` that leaves a value of the subject's type unmatched.
pub const E0501_NON_EXHAUSTIVE_MATCH: str = "E0501"

#: A pattern binding a different number of elements than the subject has.
pub const E0502_PATTERN_ARITY: str = "E0502"

#: One name bound twice by one pattern.
pub const E0503_DUPLICATE_BINDING_IN_PATTERN: str = "E0503"

#: An annotated function that can fall off its end.
pub const E0601_MISSING_RETURN: str = "E0601"

#: A store into a field the class does not declare.
pub const E0702_UNDECLARED_FIELD: str = "E0702"

#: How many names one pattern may bind before duplicates stop being reported.
#:
#: `MAX_BINDS` in `src/sema/check_stmt.c`. The C's caps are stack array sizes,
#: but every one of them is observable: past the cap the C stops reporting, so a
#: port that kept going would produce diagnostics the C does not.
pub const MAX_BINDS: int = 64

#: How many distinct literal arms a `match` tracks for `E0500`.
pub const MAX_TRACKED_LITERALS: int = 64

#: How many enum variants a `match` tracks for `E0500` and `E0501`.
pub const MAX_TRACKED_VARIANTS: int = 256

#: How many sequence lengths a `match` tracks, the width of the C's bitset.
pub const MAX_TRACKED_LENGTHS: int = 64
