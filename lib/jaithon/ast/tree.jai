#: What a caller asks of a tree that is already built: the recursive walk and
#: its two conveniences, and the two predicates the checker leans on -- whether
#: a node may appear on the left of `=`, and whether it always transfers
#: control instead of falling through.
#:
#: Everything here consumes a `Node` and constructs none, which is what lets it
#: sit downstream of `node.jai`. `children_of` is on the other side of that
#: line only because `Node.children()` calls it.
#:
#: `_flag`, `_text_at` and `_type_at` are the record accessors that came over
#: alongside `_node_at`, which `always_diverges` uses; the other three have had
#: no caller for some time. They moved as a group so that this file reads as a
#: move and nothing else. Deleting them is a separate decision from splitting
#: the file, and belongs in its own commit.
from jaithon.ast.kinds import NodeKind
from jaithon.ast.node import Node, children_of
from jaithon.ast.types import TypeExpr

fn _flag(record: dict[str, any], name: str) -> bool {
    let value: bool = record.get(name, false)
    return value
}

fn _text_at(record: dict[str, any], name: str) -> str? {
    let value: str? = record.get(name, null)
    return value
}

fn _node_at(record: dict[str, any], name: str) -> Node? {
    let value: Node? = record.get(name, null)
    return value
}

fn _type_at(record: dict[str, any], name: str) -> TypeExpr? {
    let value: TypeExpr? = record.get(name, null)
    return value
}

#: Walk the tree rooted at `node`, mirroring `jaiAstWalk`.
#:
#: `pre` runs on a node before its children and prunes them by returning false;
#: `post` runs afterwards either way, so a visitor can keep balanced enter and
#: leave state across a pruned subtree. A pre-hook that never prunes returns
#: true unconditionally.
#:
#: ```
#: var seen = 0
#: walk(Node.call(Node.ident("f"), [Node.ident("x")]), fn(node: Node) -> bool {
#:     seen += 1
#:     return node.kind != NodeKind.Call
#: }, fn(_node: Node) -> void {})
#: seen   # 1
#: ```
pub fn walk(node: Node, pre: fn(Node) -> bool, post: fn(Node) -> void) -> void {
    if pre(node) {
        for child in children_of(node) { walk(child, pre, post) }
    }
    post(node)
}

#: Every node of the tree in pre-order, root first.
#:
#: ```
#: preorder(Node.call(Node.ident("f"), [Node.ident("x")])).len()   # 3
#: ```
pub fn preorder(node: Node) -> list[Node] {
    let out: list[Node] = []
    walk(node, fn(current: Node) -> bool {
        out.push(current)
        return true
    }, fn(_current: Node) -> void {})
    return out
}

#: Every node of the tree for which `predicate` holds, in pre-order.
#:
#: ```
#: let tree = Node.call(Node.ident("f"), [Node.ident("x")])
#: find_all(tree, fn(node: Node) -> bool { return node.kind == NodeKind.Ident }).len()   # 2
#: ```
pub fn find_all(node: Node, predicate: fn(Node) -> bool) -> list[Node] {
    return [found for found in preorder(node) if predicate(found)]
}

#: Whether `node` may appear on the left of `=`.
#:
#: ```
#: is_assignable(Node.member(Node.ident("a"), "b"))   # true
#: is_assignable(Node.int_lit(1))                     # false
#: ```
pub fn is_assignable(node: Node) -> bool {
    match node.kind {
        NodeKind.Ident | NodeKind.Member | NodeKind.Index | NodeKind.Slice | NodeKind.PatBind | NodeKind.PatWildcard => {
            return true
        }
        NodeKind.OptMember => { return false }
        NodeKind.TupleLit | NodeKind.ListLit => { return _all_assignable(node.child_list("items")) }
        NodeKind.PatTuple | NodeKind.PatList => { return _all_assignable(node.child_list("elems")) }
        _ => { return false }
    }
}

fn _all_assignable(elements: list[Node]) -> bool {
    if elements.len() == 0 { return false }
    for element in elements {
        if not is_assignable(element) { return false }
    }
    return true
}

#: Whether `node` always transfers control instead of falling through.
#:
#: Conservative in the direction that keeps the missing-return check honest: a
#: loop that any `break` can leave does not diverge, whether or not that break
#: is reachable.
#:
#: ```
#: always_diverges(Node.return_stmt())                     # true
#: always_diverges(Node.expr_stmt(Node.int_lit(1)))        # false
#: ```
pub fn always_diverges(node: Node) -> bool {
    match node.kind {
        NodeKind.Return | NodeKind.Throw | NodeKind.ThrowExpr | NodeKind.Break | NodeKind.Continue => {
            return true
        }
        NodeKind.Block | NodeKind.Program => {
            for stmt in node.child_list("stmts") {
                if always_diverges(stmt) { return true }
            }
            return false
        }
        NodeKind.ExprStmt => { return always_diverges(node.require("expr")) }
        NodeKind.If | NodeKind.IfExpr => {
            let alternative = node.child("elseBranch")
            if alternative is null { return false }
            return always_diverges(node.require("thenBranch")) and always_diverges(alternative)
        }
        NodeKind.Loop => { return not _loop_has_break(node) }
        NodeKind.While => {
            let cond = node.child("cond")
            if cond is null or cond.kind != NodeKind.BoolLit { return false }
            if not cond.flag("boolLit") { return false }
            return not _loop_has_break(node)
        }
        NodeKind.For => { return false }
        NodeKind.Match | NodeKind.MatchExpr => {
            let arms = node.records("arms")
            if arms.len() == 0 { return false }
            for current in arms {
                if not always_diverges(_node_at(current, "body") ?? Node.null_lit()) {
                    return false
                }
            }
            return true
        }
        NodeKind.Try => {
            let ensure = node.child("finallyBlock")
            if ensure is not null and always_diverges(ensure) { return true }
            if not always_diverges(node.require("body")) { return false }
            for handler in node.records("catches") {
                if not always_diverges(_node_at(handler, "body") ?? Node.null_lit()) {
                    return false
                }
            }
            return true
        }
        _ => { return false }
    }
}

fn _is_loop_kind(kind: NodeKind) -> bool {
    return kind == NodeKind.While or kind == NodeKind.Loop or kind == NodeKind.For
}

fn _is_function_kind(kind: NodeKind) -> bool {
    return kind == NodeKind.FnDecl or kind == NodeKind.Lambda or kind == NodeKind.AnonFn
}

fn _loop_has_break(node: Node) -> bool {
    let label = node.opt_text("label")
    let body = node.child("body")
    if body is null { return false }
    var depth = 0
    var found = false

    fn enter(current: Node) -> bool {
        if _is_function_kind(current.kind) or
            current.kind == NodeKind.ClassDecl or
            current.kind == NodeKind.TraitDecl or
            current.kind == NodeKind.EnumDecl {
            return false
        }
        if _is_loop_kind(current.kind) {
            depth += 1
            return not found
        }
        if current.kind == NodeKind.Break {
            let target = current.opt_text("label")
            if target is null {
                if depth == 0 { found = true }
            } elif label is not null and target == label {
                found = true
            }
        }
        return not found
    }

    fn leave(current: Node) -> void {
        if _is_loop_kind(current.kind) { depth -= 1 }
    }

    walk(body, enter, leave)
    return found
}
