#: The questions the pass asks that need no scope to answer: whether a symbol
#: kind names a type, whether a symbol came from an import, whether a kind gets
#: a frame slot, what an `import a.b.c` binds, and which words are type spellings
#: the resolver never declared.
#:
#: Separated because everything here is a pure function of its argument, so it
#: is the one module of `resolve/` that can be read, and trusted, without
#: knowing anything about the pass. It is also what breaks the tie in
#: `_is_generic_instantiation`: `Box[int]` is an index or a type application
#: depending only on what `Box` turned out to be.
from jaithon.ast import Node, NodeKind
from jaithon.compile.symbol import Symbol, SymbolKind

pub let _TYPE_WORDS: set[str] = {
    "any",
    "void",
    "null",
    "bool",
    "int",
    "float",
    "str",
    "bytes",
    "range",
    "never",
    "list",
    "set",
    "dict",
    "tuple",
}

pub fn _names_a_type(kind: SymbolKind) -> bool {
    return kind == SymbolKind.Class or
        kind == SymbolKind.Trait or
        kind == SymbolKind.Enum or
        kind == SymbolKind.TypeAlias or
        kind == SymbolKind.GenericParam
}

pub fn _is_import_binding(symbol: Symbol?) -> bool {
    if symbol == null { return false }
    if symbol.kind == SymbolKind.Module { return true }
    let decl = symbol.decl
    if decl == null { return false }
    return decl.kind == NodeKind.Import or decl.kind == NodeKind.FromImport
}

pub fn _kind_is_slotted(kind: SymbolKind) -> bool {
    return kind == SymbolKind.Local or kind == SymbolKind.Param
}

pub fn _import_binding_name(node: Node) -> str? {
    let path = node.opt_text("alias") ?? node.opt_text("path")
    if path == null { return null }
    let parts = path.split(".")
    return parts[parts.len() - 1]
}
