#: The parser's positional node builders against the named form they replaced.
#:
#: `Node.of_slots` takes a slot list in schema order and checks only its
#: length, so a builder that put two fields in the wrong order would build a
#: tree every later pass misreads. Each builder here is compared slot by slot
#: with `Node(kind, {named fields}, span)` over the same parts.
from std.test import assert_eq, assert_throws, assert_true
from jaithon.ast import DeclKind, Node, NodeKind, OpKind, Span, Visibility
from jaithon.compile.parse.node_build import _binary, _call, _ident, _int_lit, _member, _str_lit
from jaithon.compile.parse.node_build import _assign, _block, _bool_lit, _expr_stmt, _if, _index
from jaithon.compile.parse.node_build import _node, _pat_bind, _pat_class, _pat_seq, _return
from jaithon.compile.parse.node_build import _sequence, _unary, _var_decl

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

fn test_patterns_and_sequences() {
    let span = Span(0, 1, 2)
    let item = Node.ident("i")
    let items: list[Node] = [item]
    for kind in [NodeKind.ListLit, NodeKind.SetLit, NodeKind.TupleLit] {
        _same(_sequence(kind, span, items), Node(kind, {"items": items}, span))
    }
    _same(_pat_bind(span, "x"), Node(NodeKind.PatBind, {"name": "x", "type": null}, span))
    for kind in [NodeKind.PatTuple, NodeKind.PatList, NodeKind.PatOr] {
        _same(_pat_seq(kind, span, items, 2), Node(kind, {"elems": items, "restIndex": 2}, span))
    }
    let names: list[str?] = ["a", null]
    for kind in [NodeKind.PatClass, NodeKind.PatEnum] {
        let named = Node(
            kind,
            {
                "typeName": "T",
                "variantName": "V",
                "fieldNames": names,
                "subPatterns": items,
            },
            span
        )
        _same(_pat_class(kind, span, "T", "V", names, items), named)
    }
}

fn test_if_assign_and_declarations() {
    let span = Span(0, 4, 9)
    let cond = Node.ident("c")
    let then_branch = Node.ident("t")
    let else_branch = Node.ident("e")
    for kind in [NodeKind.If, NodeKind.IfExpr] {
        let named = Node(
            kind,
            {"cond": cond, "thenBranch": then_branch, "elseBranch": else_branch},
            span
        )
        _same(_if(kind, span, cond, then_branch, else_branch), named)
    }
    let target = Node.ident("x")
    let named_assign = Node(
        NodeKind.Assign,
        {"target": target, "value": cond, "op": OpKind.Sub, "isCompound": true},
        span
    )
    _same(_assign(span, target, cond, OpKind.Sub, true), named_assign)
    let named_decl = Node(
        NodeKind.VarDecl,
        {
            "varDeclKind": DeclKind.Var,
            "pattern": target,
            "declaredType": null,
            "init": cond,
            "visibility": Visibility.Public,
        },
        span
    )
    _same(_var_decl(span, DeclKind.Var, target, null, cond, Visibility.Public), named_decl)
    _same(_bool_lit(span, true), Node(NodeKind.BoolLit, {"boolLit": true}, span))
}
