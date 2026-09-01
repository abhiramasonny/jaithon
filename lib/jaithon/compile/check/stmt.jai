#: Statements, control flow, and the narrowing a branch is allowed to keep.
#:
#: A port of `src/sema/check_stmt.c`: patterns and `match`, the `let`/`var`/
#: `const` that appear inside a body, assignment, blocks and loops, `return`,
#: `throw`, `try`, `defer`, `assert`, and function bodies.
#:
#: Facade: imports and re-exports the modules of the `stmt/` directory. The
#: surface is the twenty-five names it was before the split, so `checker.jai`
#: and the five `tests/checker/test_check_stmt*` files reach it unchanged.
#:
#: `Ctx` is the seam and `expr.jai` owns it; `install` — in `stmt/function.jai`
#: — fills in five of its callbacks. `checker.jai` calls `install` too and then
#: replaces `Ctx.check_decl`, `Ctx.resolve_ast_type` and `Ctx.layout_decl`, none
#: of which this directory can supply. Until it lands, every written annotation
#: here reads as `any`, a declaration inside a block is unchecked, and a class
#: is never laid out.
#:
#: Two gaps are the AST's rather than a callback's: a `for` loop's hidden
#: iterator binding is never typed, because the Jaithon `For` schema has no such
#: field; and a `const`'s folded value lives in `Ctx.consts` keyed by **name**
#: rather than on the `Symbol`, so two `const K` in different scopes collide.
#:
#: ## What lives where
#:
#: The directory is a straight line — each file imports only the files above it
#: — so there is no seam inside it and nothing to wire. `codes.jai` is the
#: bottom and imports nothing: every diagnostic a statement can raise and every
#: cap it counts within, gathered so that the list can be read against
#: `src/common/diag.h` in one place. `fields.jai` is the other leaf: the
#: flattened field list of a class or enum, which three unrelated checks index.
#:
#: On top of those, `pattern.jai` binds one pattern against the type it
#: destructures. `matching.jai` runs the arms of a `match` over it, accumulates
#: coverage across them and answers the type of the whole form; it is spelled
#: `matching` because `match` is a keyword and a keyword cannot be a segment of
#: a module path. `store.jai` holds the declarations and every form of
#: assignment, the two places a value is put somewhere.
#:
#: `flow.jai` sits above all of those and holds `stmt`, the dispatcher, along
#: with blocks, branches, loops and the jump forms. `function.jai` is last: a
#: signature, a body checked against it, and `install`, which is the only place
#: that names every entry point at once.
from jaithon.compile.check.stmt.codes import E0301_ASSIGN_TO_IMMUTABLE
from jaithon.compile.check.stmt.codes import E0303_NON_CONSTANT_CONST, E0304_MISSING_INITIALIZER
from jaithon.compile.check.stmt.codes import E0402_UNKNOWN_TYPE, E0408_RETURN_TYPE_MISMATCH
from jaithon.compile.check.stmt.codes import E0500_UNREACHABLE_ARM, E0501_NON_EXHAUSTIVE_MATCH
from jaithon.compile.check.stmt.codes import E0502_PATTERN_ARITY
from jaithon.compile.check.stmt.codes import E0503_DUPLICATE_BINDING_IN_PATTERN
from jaithon.compile.check.stmt.codes import E0601_MISSING_RETURN, E0702_UNDECLARED_FIELD
from jaithon.compile.check.stmt.codes import MAX_BINDS, MAX_TRACKED_LENGTHS
from jaithon.compile.check.stmt.codes import MAX_TRACKED_LITERALS, MAX_TRACKED_VARIANTS
from jaithon.compile.check.stmt.pattern import pattern
from jaithon.compile.check.stmt.matching import match_expr
from jaithon.compile.check.stmt.flow import block, stmt
from jaithon.compile.check.stmt.function import declared_return_type, function, function_type
from jaithon.compile.check.stmt.function import install, is_init_method, push_generics

export { E0301_ASSIGN_TO_IMMUTABLE }
export { E0303_NON_CONSTANT_CONST, E0304_MISSING_INITIALIZER }
export { E0402_UNKNOWN_TYPE, E0408_RETURN_TYPE_MISMATCH }
export { E0500_UNREACHABLE_ARM, E0501_NON_EXHAUSTIVE_MATCH }
export { E0502_PATTERN_ARITY }
export { E0503_DUPLICATE_BINDING_IN_PATTERN }
export { E0601_MISSING_RETURN, E0702_UNDECLARED_FIELD }
export { MAX_BINDS, MAX_TRACKED_LENGTHS }
export { MAX_TRACKED_LITERALS, MAX_TRACKED_VARIANTS }
export { pattern }
export { match_expr }
export { block, stmt }
export { declared_return_type, function, function_type }
export { install, is_init_method, push_generics }
