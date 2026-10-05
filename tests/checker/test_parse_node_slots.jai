#: The parser's positional node builders against the named form they replaced.
#:
#: `Node.of_slots` takes a slot list in schema order and checks only its
#: length, so a builder that put two fields in the wrong order would build a
#: tree every later pass misreads. Each builder here is compared slot by slot
#: with `Node(kind, {named fields}, span)` over the same parts.
from std.test import assert_eq, assert_throws, assert_true
from jaithon.ast import Node, NodeKind, OpKind, Span
from jaithon.compile.parse.node_build import _binary, _call, _ident, _int_lit, _member, _str_lit
from jaithon.compile.parse.node_build import _block, _expr_stmt, _index, _node, _return, _unary

fn _same(built: Node, named: Node) -> void {
    assert_eq(built.kind, named.kind)
    assert_eq(built.fields.len(), named.fields.len())
    for at in 0..named.fields.len() {
        assert_true(built.fields[at] == named.fields[at], f"slot {at} of {built.kind}")
    }
}

fn test_ident_and_literals() {
    let span = Span(0, 3, 7)
    _same(_ident(span, "name"), Node(NodeKind.Ident, {"name": "name"}, span))
    _same(_str_lit(span, "text"), Node(NodeKind.StrLit, {"chars": "text"}, span))
    _same(_int_lit(span, 42), Node(NodeKind.IntLit, {"intLit": 42}, span))
}

fn test_operators() {
    let span = Span(0, 0, 5)
    let left = Node.ident("a")
    let right = Node.ident("b")
    _same(
        _unary(span, OpKind.Neg, left),
        Node(NodeKind.Unary, {"op": OpKind.Neg, "operand": left}, span)
    )
    for kind in [NodeKind.Binary, NodeKind.Logical] {
        let named = Node(kind, {"op": OpKind.And, "left": left, "right": right}, Span.none())
        _same(_binary(kind, OpKind.And, left, right), named)
    }
}

fn test_member_and_call() {
    let span = Span(0, 1, 9)
    let object = Node.ident("obj")
    for kind in [NodeKind.Member, NodeKind.OptMember] {
        _same(
            _member(kind, span, object, "field"),
            Node(kind, {"object": object, "name": "field"}, span)
        )
    }
    let args: list[dict[str, any]] = []
    _same(_call(span, object, args), Node(NodeKind.Call, {"callee": object, "args": args}, span))
}

fn test_a_short_slot_list_is_refused() {
    assert_throws(ValueError, || Node.of_slots(NodeKind.Binary, Span.none(), [OpKind.Add]))
}

fn test_statements_and_index() {
    let span = Span(0, 2, 6)
    let value = Node.ident("v")
    let stmts: list[Node] = [Node.ident("s")]
    _same(_block(span, stmts), Node(NodeKind.Block, {"stmts": stmts}, span))
    _same(_expr_stmt(span, value), Node(NodeKind.ExprStmt, {"expr": value}, span))
    _same(_return(span, value), Node(NodeKind.Return, {"value": value}, span))
    _same(_return(span, null), Node(NodeKind.Return, {"value": null}, span))
    let object = Node.ident("xs")
    _same(
        _index(span, object, value),
        Node(NodeKind.Index, {"object": object, "index": value}, span)
    )
}

fn test_an_empty_overlay_builds_the_defaults() {
    let span = Span(0, 0, 4)
    _same(_node(NodeKind.Block, span), Node(NodeKind.Block, null, span))
    _same(_node(NodeKind.NullLit, span, {}), Node(NodeKind.NullLit, null, span))
}
