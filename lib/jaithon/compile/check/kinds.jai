#: The twenty-one semantic type kinds, and the ordinal that orders them. The
#: enum order is a wire format: it must match `TY_*` in `src/sema/types.h`.

#: What a type is, once the checker has decided. `Alias` is here for ordinal
#: fidelity only: nothing in either front end constructs one.
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
pub fn ordinal(kind: TyKind) -> int {
    return _ORDINAL[kind]
}

#: How the kinds that are their own spelling render. `Never` has no source
#: spelling; the C's `never` is here so diagnostics can name the bottom type.
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
pub const JOIN_MAX_MEMBERS: int = 8

#: How far alias resolution will walk before deciding it found a cycle.
pub const MAX_ALIAS_HOPS: int = 32

#: How deep a declaration walk goes before it stops.
pub const MAX_DECL_DEPTH: int = 32

#: How deep a type renders before the rest becomes `...`. The `>` guard runs
#: from depth 0, so the first `...` is two levels late, as in the C.
pub const MAX_RENDER_DEPTH: int = 32
