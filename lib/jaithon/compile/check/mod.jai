#: The semantic type system: the Jaithon half of `src/sema`.
#:
#: `parse -> resolve -> emit` is what `jaithon.compile.compile_source` runs
#: today. This directory is the stage that goes between the last two, and the
#: reason the self-hosted front end cannot yet produce the same bytecode the C
#: one does: without types there is no `OP_TYPE_GUARD` at an `any` boundary, no
#: `OP_TO_FLOAT` where an int is stored as a float, and no constant folding —
#: which together account for about half the divergences `make bootstrap`
#: reports.
#:
#: Phase 1 is this type system. Phase 2 is the checker that computes with it.
#: Nothing here is wired into the pipeline yet, deliberately: a half-built
#: checker would move the divergence count for reasons that are not progress.
#:
#: The files, in dependency order:
#:
#: | file | what it holds |
#: |---|---|
#: | `kinds` | the 21 kinds and the **ordinal** that orders union members |
#: | `ty` | `Type`, `TypeDecl`, `FieldInfo`, `MemberInfo` — data only |
#: | `render` | the canonical type string, byte-exact against the C |
#: | `universe` | the intern table, the primitives, the constructors |
#: | `union` | flatten, absorb, drop, sort, dedup, collapse |
#: | `nominal` | the class, trait and enum hierarchy |
#: | `assign` | assignability and widening — what decides a guard |
#: | `relate` | equality, subtyping, join, narrowing |
#: | `predicate` | numeric, optional, callable, iterable, indexable |
#: | `operator` | the result type of `a op b` and of `op a` |
#: | `substitute` | generic parameter replacement |
#: | `dump` | the Phase 0 sema-dump reader and differ |
#:
#: Two properties are load-bearing and easy to break by accident.
#:
#: **Identity is interning.** Two structurally equal types must be the same
#: object, because the C gets that free from an arena and every relation here
#: assumes a comparison rather than a walk. Build types through `Universe`, not
#: through `Type` directly.
#:
#: **The rendered string is a wire format.** `make sema-diff` compares rendered
#: types between the two front ends, so the union sort order — and therefore
#: `kinds.ORDINAL` — decides whether a program diffs clean.

#: There is deliberately nothing else in this file.
#:
#: Jaithon does not re-export a name a module merely imported, so a package
#: module cannot forward its siblings' surface the way `__init__.py` does
#: elsewhere. Importing each name here would only raise `unused import` on
#: every one of them and promise a `from jaithon.compile.check import Type`
#: that would not work. Import the file that owns the name:
#:
#: ```text
#: from jaithon.compile.check.ty import Type
#: from jaithon.compile.check.universe import Universe, list_of
#: from jaithon.compile.check.assign import assignable
#: ```
