#: The twenty-one semantic type kinds, and the ordinal that orders them.
#:
#: This is the syntactic layer's counterpart, not its replacement.
#: `jaithon.ast.TypeKind` has seven members and describes how a type was
#: *written* — `Name`, `Generic`, `Optional`, `Union`, `Fn`, `Tuple`, `Infer`.
#: `TyKind` here describes what a type *is* after the checker has worked it out,
#: and the checker imports both. They are separate names because they are
#: separate things and a file that needs one usually needs the other.
#:
#: **The order of this enum is a wire format.** `ORDINAL` must equal the
#: position of the matching `TY_*` in `src/sema/types.h`, because union members
#: are sorted by it, the sort decides the rendered union string, and the
#: rendered string is the value `make sema-diff` compares between the two front
#: ends. Reordering the enum changes what every union renders as, silently, and
#: the symptom is a diff on files that have nothing to do with the change.
#:
#: ```
#: ordinal(TyKind.Any)      # 0
#: ordinal(TyKind.Int)      # 4
#: ordinal(TyKind.Alias)    # 20
#: ```

#: What a type is, once the checker has decided.
#:
#: `Alias` is here for ordinal fidelity only. Nothing in either front end
#: constructs one: the sole producer of named types never emits this kind, so an
#: alias is resolved where it is written and never reaches the type universe.
#: Porting the kind is required; porting a resolution algorithm for it is not.
pub enum TyKind {
    Any,
    Never,
    Null,
    Bool,
    Int,
    Float,
    Str,
    Bytes,
    List,
    Dict,
    Set,
    Tuple,
    Range,
    Fn,
    Class,
    Trait,
    Enum,
    Module,
    Union,
    GenericParam,
    Alias,
}

#: Every kind, in the order `src/sema/types.h` declares them.
#:
#: The list is the definition and `ORDINAL` is derived from it, so the two
#: cannot drift apart.
pub let KINDS: list[TyKind] = [
    TyKind.Any,
    TyKind.Never,
    TyKind.Null,
    TyKind.Bool,
    TyKind.Int,
    TyKind.Float,
    TyKind.Str,
    TyKind.Bytes,
    TyKind.List,
    TyKind.Dict,
    TyKind.Set,
    TyKind.Tuple,
    TyKind.Range,
    TyKind.Fn,
    TyKind.Class,
    TyKind.Trait,
    TyKind.Enum,
    TyKind.Module,
    TyKind.Union,
    TyKind.GenericParam,
    TyKind.Alias,
]

fn _build_ordinals() -> dict[TyKind, int] {
    var out: dict[TyKind, int] = {}
    for (position, kind) in KINDS.enumerate() {
        out[kind] = position
    }
    return out
}

let _ORDINAL: dict[TyKind, int] = _build_ordinals()

#: Where `kind` sits in the C enum, which is the key unions sort on.
#:
#: ```
#: ordinal(TyKind.Bool) < ordinal(TyKind.Int)      # true
#: ordinal(TyKind.Alias)                           # 20
#: ```
pub fn ordinal(kind: TyKind) -> int {
    return _ORDINAL[kind]
}

#: How the eight kinds that are their own spelling render.
#:
#: Every other kind renders from its parts (`list[int]`) or its name (`Box`),
#: so it is not here. `Never` is absent deliberately: it has no source spelling
#: and the C renders it as `never`, which this table carries so that a
#: diagnostic naming the bottom type says something.
pub let PRIMITIVE_SPELLING: dict[TyKind, str] = {
    TyKind.Any: "any",
    TyKind.Never: "never",
    TyKind.Null: "null",
    TyKind.Bool: "bool",
    TyKind.Int: "int",
    TyKind.Float: "float",
    TyKind.Str: "str",
    TyKind.Bytes: "bytes",
    TyKind.Range: "range",
}

#: `fn(a, b, ...rest)` — the last positional parameter collects the rest.
pub const FN_VARIADIC: int = 1

#: `fn(a, **kwargs)` — the last parameter collects keyword arguments.
pub const FN_KWREST: int = 2

#: Past this many members a join gives up and answers `any`.
#:
#: A least upper bound that keeps growing is worse than no answer: it makes
#: every later comparison slower and the type it prints is unreadable.
pub const JOIN_MAX_MEMBERS: int = 8

#: How far alias resolution will walk before deciding it found a cycle.
pub const MAX_ALIAS_HOPS: int = 32

#: How deep a declaration walk goes before it stops.
pub const MAX_DECL_DEPTH: int = 32

#: How deep a type renders before the rest becomes `...`.
#:
#: The top level is depth 0 and the guard is `depth > MAX_RENDER_DEPTH`, so the
#: first `...` appears at the thirty-fourth level rather than the
#: thirty-second. That off-by-two is the C's and is reproduced on purpose.
pub const MAX_RENDER_DEPTH: int = 32
