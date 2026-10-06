#: The constructors that turn already-parsed parts into an `ast` node, a
#: `TypeExpr`, or one of the plain `dict[str, any]` records the tree carries
#: for parameters, arguments, match arms, comprehension clauses, catch clauses,
#: fields, variants, generics and import items.
#:
#: Not one of these takes a `Parser`, reads a token or reports a diagnostic.
#: They are the shape of the tree and nothing else, which is why this file sits
#: below every other module of `parse/` and can be read on its own.
from jaithon.ast import DeclKind, Node, NodeKind, OpKind, Span, TypeExpr, TypeKind, Visibility

pub fn _node(kind: NodeKind, span: Span, fields: dict[str, any] = {}) -> Node {
    # An empty overlay is no overlay: `init` skips it for null, not for `{}`.
    if fields.len() == 0 { return Node(kind, null, span) }
    return Node(kind, fields, span)
}

pub fn _type(
    kind: TypeKind,
    span: Span,
    name: str? = null,
    args: list[TypeExpr] = [],
    inner: TypeExpr? = null
) -> TypeExpr {
    return TypeExpr(kind, name, args, inner, span)
}

pub fn _join(first: Span, second: Span) -> Span {
    if first.file < 0 { return second }
    if second.file < 0 { return first }
    if first.file != second.file { return first }
    let start = min(first.start, second.start)
    return Span(first.file, start, max(start, max(first.end, second.end)))
}

# The builders below that a parse calls once per node build their slot list
# in schema order (`Node.of_slots`) instead of naming fields in a dict; each is
# pinned against the named form by tests/checker/test_parse_node_slots.jai.
pub fn _ident(span: Span, name: str) -> Node {
    return Node.of_slots(NodeKind.Ident, span, [name])
}

pub fn _str_lit(span: Span, chars: str) -> Node {
    # `length` stays at its default, 0, as it did when this named `chars` alone.
    return Node.of_slots(NodeKind.StrLit, span, [chars, 0])
}

pub fn _sequence(kind: NodeKind, span: Span, items: list[Node]) -> Node {
    # `ListLit`, `SetLit` and `TupleLit` share one schema: items.
    return Node.of_slots(kind, span, [items])
}

pub fn _dict_lit(span: Span, keys: list[Node], values: list[Node]) -> Node {
    return _node(NodeKind.DictLit, span, {"keys": keys, "values": values})
}

pub fn _unary(span: Span, op: OpKind, operand: Node) -> Node {
    return Node.of_slots(NodeKind.Unary, span, [op, operand])
}

pub fn _binary(kind: NodeKind, op: OpKind, left: Node, right: Node) -> Node {
    # `Binary` and `Logical` share one schema: op, left, right.
    return Node.of_slots(kind, _join(left.span, right.span), [op, left, right])
}

pub fn _wrap(kind: NodeKind, span: Span, operand: Node) -> Node {
    return _node(kind, span, {"operand": operand})
}

pub fn _member(kind: NodeKind, span: Span, object: Node, name: str) -> Node {
    # `Member` and `OptMember` share one schema: object, name.
    return Node.of_slots(kind, span, [object, name])
}

pub fn _call(span: Span, callee: Node, args: list[dict[str, any]]) -> Node {
    return Node.of_slots(NodeKind.Call, span, [callee, args])
}

pub fn _int_lit(span: Span, value: any) -> Node {
    return Node.of_slots(NodeKind.IntLit, span, [value])
}

pub fn _index(span: Span, object: Node, index: Node) -> Node {
    return Node.of_slots(NodeKind.Index, span, [object, index])
}

pub fn _block(span: Span, stmts: list[Node]) -> Node {
    return Node.of_slots(NodeKind.Block, span, [stmts])
}

pub fn _expr_stmt(span: Span, expr: Node) -> Node {
    return Node.of_slots(NodeKind.ExprStmt, span, [expr])
}

pub fn _return(span: Span, value: Node?) -> Node {
    return Node.of_slots(NodeKind.Return, span, [value])
}

pub fn _bool_lit(span: Span, value: bool) -> Node {
    return Node.of_slots(NodeKind.BoolLit, span, [value])
}

pub fn _if(kind: NodeKind, span: Span, cond: Node, then_branch: Node, else_branch: Node?) -> Node {
    # `If` and `IfExpr` share one schema: cond, thenBranch, elseBranch.
    return Node.of_slots(kind, span, [cond, then_branch, else_branch])
}

pub fn _assign(span: Span, target: Node, value: Node, op: OpKind, is_compound: bool) -> Node {
    return Node.of_slots(NodeKind.Assign, span, [target, value, op, is_compound])
}

pub fn _var_decl(
    span: Span,
    decl_kind: DeclKind,
    pattern: Node,
    declared: TypeExpr?,
    init: Node?,
    visibility: Visibility
) -> Node {
    return Node.of_slots(NodeKind.VarDecl, span, [decl_kind, pattern, declared, init, visibility])
}

pub fn _match_node(kind: NodeKind, span: Span, subject: Node, arms: list[dict[str, any]]) -> Node {
    return _node(kind, span, {"subject": subject, "arms": arms})
}

pub fn _pat_bind(span: Span, name: str?) -> Node {
    return Node.of_slots(NodeKind.PatBind, span, [name, null])
}

pub fn _pat_seq(kind: NodeKind, span: Span, elems: list[Node], rest_index: int) -> Node {
    # `PatTuple`, `PatList` and `PatOr` share one schema: elems, restIndex.
    return Node.of_slots(kind, span, [elems, rest_index])
}

pub fn _pat_class(
    kind: NodeKind,
    span: Span,
    type_name: str,
    variant_name: str?,
    field_names: list[str?],
    sub_patterns: list[Node]
) -> Node {
    # `PatClass` and `PatEnum` share one schema.
    return Node.of_slots(kind, span, [type_name, variant_name, field_names, sub_patterns])
}

pub fn _nulls(count: int) -> list[str?] {
    return [null for _index in 0..count]
}

pub fn _param(
    name: str?,
    declared: TypeExpr?,
    default_value: Node?,
    span: Span,
    is_variadic: bool,
    is_kw_rest: bool
) -> dict[str, any] {
    return {
        "name": name,
        "type": declared,
        "defaultValue": default_value,
        "span": span,
        "isVariadic": is_variadic,
        "isKwRest": is_kw_rest,
    }
}

pub fn _arg(name: str?, value: Node, is_spread: bool, span: Span) -> dict[str, any] {
    return {"name": name, "value": value, "isSpread": is_spread, "span": span}
}

pub fn _arm(pattern: Node, guard: Node?, body: Node, span: Span) -> dict[str, any] {
    return {"pattern": pattern, "guard": guard, "body": body, "span": span}
}

pub fn _comp_clause(
    pattern: Node,
    iterable: Node,
    conditions: list[Node],
    span: Span
) -> dict[str, any] {
    return {"pattern": pattern, "iterable": iterable, "conditions": conditions, "span": span}
}

pub fn _catch(name: str?, types: list[TypeExpr], body: Node, span: Span) -> dict[str, any] {
    return {"name": name, "types": types, "body": body, "span": span}
}

pub fn _field(
    name: str?,
    declared: TypeExpr?,
    default_value: Node?,
    visibility: Visibility,
    is_static: bool,
    is_let: bool,
    span: Span
) -> dict[str, any] {
    return {
        "name": name,
        "type": declared,
        "defaultValue": default_value,
        "visibility": visibility,
        "isStatic": is_static,
        "isLet": is_let,
        "span": span,
    }
}

pub fn _variant(name: str?, params: list[dict[str, any]], span: Span) -> dict[str, any] {
    return {"name": name, "params": params, "span": span}
}

pub fn _generic(name: str?, bound: TypeExpr?, span: Span) -> dict[str, any] {
    return {"name": name, "bound": bound, "span": span}
}

pub fn _import_item(name: str?, alias: str?, span: Span) -> dict[str, any] {
    return {"name": name, "alias": alias, "span": span}
}
