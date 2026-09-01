#: The questions about a tree that need no `Emitter`: whether a symbol binds
#: anywhere other than a local, a parameter, an upvalue or a field, whether an
#: iterable is the `X.items()` call the lazy loop opcode wants, and the small
#: accessors that reach past a node's own spelling -- the two branches of a
#: ternary or an `if`, a class pattern's field name, the last segment of an
#: import path.
#:
#: Grouped by the fact that they only READ, which is what lets every module of
#: `emit/` call them without an ordering constraint. Nothing here writes a byte.
from jaithon.ast import Node, NodeKind
from jaithon.compile.symbol import Symbol, SymbolKind

pub fn _symbol_is_global_like(symbol: Symbol) -> bool {
    let kind = symbol.kind
    return not (kind == SymbolKind.Local or kind == SymbolKind.Param or kind == SymbolKind.Upvalue or kind == SymbolKind.Field)
}

#: Emit the iterable of a `for` loop, leaving an iterator on the stack.
#:
#: `for … in X.items()` gets `OP_GET_ITER_ITEMS` applied to X rather than a call
#: to `items()` followed by `OP_GET_ITER`. On a dict that iterates the table
#: lazily instead of building the whole list of pairs first — `jaiDictItems`
#: allocates an N-element list of N fresh 2-tuples per call and keeps all N
#: alive at once, so every collection during the loop marks the lot.
#:
#: Matched on SHAPE only, because this file has no checker types. The opcode
#: decides at run time what the receiver actually is, so a user class defining
#: its own `items()` still gets that method called and behaves exactly as
#: before.
pub fn _is_items_call(iterable: Node?) -> bool {
    if iterable == null { return false }
    let node: Node = iterable
    if node.kind != NodeKind.Call { return false }
    if node.records("args").len() != 0 { return false }
    let callee = node.child("callee")
    if callee == null { return false }
    let target: Node = callee
    if target.kind != NodeKind.Member { return false }
    if target.opt_text("name") != "items" { return false }
    let receiver = target.child("object")
    if receiver == null { return false }
    let recv: Node = receiver
    return recv.kind != NodeKind.Super
}

#: The receiver of an `X.items()` call matched by `_is_items_call`.
pub fn _items_receiver(iterable: Node?) -> Node? {
    if iterable == null { return null }
    let node: Node = iterable
    return node.child("callee")?.child("object")
}

pub fn _then_of(node: Node) -> Node? {
    if node.kind == NodeKind.Ternary { return node.child("thenExpr") }
    return node.child("thenBranch")
}

pub fn _else_of(node: Node) -> Node? {
    if node.kind == NodeKind.Ternary { return node.child("elseExpr") }
    return node.child("elseBranch")
}

pub fn _field_name(pattern: Node, index: int) -> str {
    let names: list[str?] = pattern.get("fieldNames")
    if index >= names.len() { return f"{index}" }
    return names[index] ?? f"{index}"
}

pub fn _last_segment(path: str?) -> str? {
    if path == null { return null }
    let parts = path.split(".")
    return parts[parts.len() - 1]
}
