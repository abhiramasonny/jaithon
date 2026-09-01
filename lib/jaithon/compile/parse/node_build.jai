#: The constructors that turn already-parsed parts into an `ast` node, a
#: `TypeExpr`, or one of the plain `dict[str, any]` records the tree carries
#: for parameters, arguments, match arms, comprehension clauses, catch clauses,
#: fields, variants, generics and import items.
#:
#: Not one of these takes a `Parser`, reads a token or reports a diagnostic.
#: They are the shape of the tree and nothing else, which is why this file sits
#: below every other module of `parse/` and can be read on its own.
from jaithon.ast import Node, NodeKind, OpKind, Span, TypeExpr, TypeKind, Visibility

pub fn _node(kind: NodeKind, span: Span, fields: dict[str, any] = {}) -> Node {
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

pub fn _ident(span: Span, name: str) -> Node {
    return _node(NodeKind.Ident, span, {"name": name})
}

pub fn _str_lit(span: Span, chars: str) -> Node {
    return _node(NodeKind.StrLit, span, {"chars": chars})
}

pub fn _sequence(kind: NodeKind, span: Span, items: list[Node]) -> Node {
    return _node(kind, span, {"items": items})
}

pub fn _dict_lit(span: Span, keys: list[Node], values: list[Node]) -> Node {
    return _node(NodeKind.DictLit, span, {"keys": keys, "values": values})
}

pub fn _unary(span: Span, op: OpKind, operand: Node) -> Node {
    return _node(NodeKind.Unary, span, {"op": op, "operand": operand})
}

pub fn _binary(kind: NodeKind, op: OpKind, left: Node, right: Node) -> Node {
    return _node(kind, _join(left.span, right.span), {"op": op, "left": left, "right": right})
}

pub fn _wrap(kind: NodeKind, span: Span, operand: Node) -> Node {
    return _node(kind, span, {"operand": operand})
}

pub fn _member(kind: NodeKind, span: Span, object: Node, name: str) -> Node {
    return _node(kind, span, {"object": object, "name": name})
}

pub fn _match_node(kind: NodeKind, span: Span, subject: Node, arms: list[dict[str, any]]) -> Node {
    return _node(kind, span, {"subject": subject, "arms": arms})
}

pub fn _pat_bind(span: Span, name: str?) -> Node {
    return _node(NodeKind.PatBind, span, {"name": name, "type": null})
}

pub fn _pat_seq(kind: NodeKind, span: Span, elems: list[Node], rest_index: int) -> Node {
    return _node(kind, span, {"elems": elems, "restIndex": rest_index})
}

pub fn _pat_class(
    kind: NodeKind,
    span: Span,
    type_name: str,
    variant_name: str?,
    field_names: list[str?],
    sub_patterns: list[Node]
) -> Node {
    return _node(
        kind,
        span,
        {
            "typeName": type_name,
            "variantName": variant_name,
            "fieldNames": field_names,
            "subPatterns": sub_patterns,
        }
    )
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
