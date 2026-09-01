#: Control flow, and the narrowing each shape of it is allowed to keep.
#:
#: `stmt` at the bottom of this file is the dispatcher the whole directory
#: exists to serve, and `block` is the scope every fact eventually unwinds to.
#:
#: **There is no truthiness** (spec §5.1), so every condition goes through
#: `expr.condition` and is checked for `bool` rather than coerced. And
#: **narrowing is scoped**: a fact learned by `if x is C` is pushed on entry to
#: the arm it holds in and popped on the way out, which is why so many functions
#: here are a `narrow_mark`/`narrow_restore` pair around a recursive call.
#:
#: Two narrowing rules are worth stating outright:
#:
#: - **The false edge is collected, not inverted.** `a or b` proves nothing when
#:   true, while `not (a or b)` proves both parts when false, and the second is
#:   the `if x is null or x <= 0 { return }` guard the library is written with.
#: - **A diverging `then` with no `else` narrows the rest of the block.** Those
#:   facts are deliberately not restored; `block` unwinds them.
#:
#: A gap here is the AST's rather than a check's: a `for` loop's hidden iterator
#: binding is never typed, because the Jaithon `For` schema has no such field.
#:
#: What is NOT here: the store that ends a fact. `stmt` hands a `VarDecl` or an
#: `Assign` to `store.jai`, which checks it against the declaration and calls
#: `Ctx.narrow_invalidate` itself.
from jaithon.ast import Node, NodeKind, TypeExpr
from jaithon.ast import always_diverges, is_declaration, is_expression
from jaithon.compile.check.ctx import Ctx, is_any, is_never, is_void, or_any
from jaithon.compile.check.expr import E0400_TYPE_MISMATCH, W0102_UNREACHABLE_CODE
from jaithon.compile.check.expr import condition, expr, iterable_element, value
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.predicate import is_optional
from jaithon.compile.check.render import render
from jaithon.compile.check.stmt.codes import E0408_RETURN_TYPE_MISMATCH
from jaithon.compile.check.stmt.matching import match_expr
from jaithon.compile.check.stmt.pattern import pattern
from jaithon.compile.check.stmt.store import _check_assign, _check_var_decl
from jaithon.compile.check.ty import Type
from jaithon.compile.check.union import union_of

#: Check every statement of a block, warning once about the first dead one.
#:
#: `jaiChkBlock`, and the shape of the installed `Ctx.check_block`. The mark and
#: restore around the loop are what let `if x == null { return }` narrow the
#: statements that follow it and no further: `_check_if` leaves those facts
#: applied deliberately, and this is where they come off.
pub fn block(c: Ctx, node: Node?) -> void {
    if node is null { return }
    let mark = c.narrow_mark()
    var diverged = false
    var warned = false

    for statement in node.child_list("stmts") {
        if diverged and not warned and not is_declaration(statement.kind) {
            c.warn(
                W0102_UNREACHABLE_CODE,
                statement.span,
                "unreachable code: the previous statement always transfers control"
            )
            warned = true
        }
        stmt(c, statement)
        if not diverged and always_diverges(statement) { diverged = true }
    }

    c.narrow_restore(mark)
}

fn _check_if(c: Ctx, node: Node) -> void {
    let cond = node.child("cond")
    condition(c, cond, "an `if`")

    let facts = c.collect_facts(cond, false)
    let else_facts = c.collect_facts(cond, true)

    let then_branch = node.child("thenBranch")
    var mark = c.narrow_mark()
    c.narrow_apply(facts, false)
    if then_branch is not null { stmt(c, then_branch) }
    c.narrow_restore(mark)

    let else_branch = node.child("elseBranch")
    if else_branch is not null {
        mark = c.narrow_mark()
        c.narrow_apply(else_facts, false)
        stmt(c, else_branch)
        c.narrow_restore(mark)
        return
    }

    if then_branch is not null and always_diverges(then_branch) {
        c.narrow_apply(else_facts, false)
    }
}

fn _check_loop(c: Ctx, node: Node) -> void {
    let body = node.child("body")
    let cond = node.child("cond")
    if node.kind == NodeKind.While and cond is not null {
        condition(c, cond, "a `while`")
        let facts = c.collect_facts(cond, false)
        let mark = c.narrow_mark()
        c.narrow_apply(facts, false)
        if body is not null { stmt(c, body) }
        c.narrow_restore(mark)
        return
    }
    if body is not null { stmt(c, body) }
}

fn _check_for(c: Ctx, node: Node) -> void {
    let over = node.child("iterable")
    let source = value(c, over)
    let elem = iterable_element(c, over, source)

    let mark = c.narrow_mark()
    pattern(c, node.child("pattern"), elem)

    let body = node.child("body")
    if body is not null { stmt(c, body) }
    c.narrow_restore(mark)
}

fn _check_return(c: Ctx, node: Node) -> void {
    let want = c.current_return_type
    let returned = node.child("value")

    if returned is null {
        if want is not null and
            not is_any(want) and
            not is_void(c, want) and
            want.kind != TyKind.Null and
            not is_optional(want) {
            let expected = render(want)
            let d = c.error(
                E0408_RETURN_TYPE_MISMATCH,
                node.span,
                f"this function must return `{expected}`"
            )
            d.with_help(f"add a value: `return <{expected}>`")
        }
        return
    }

    if want is not null and is_void(c, want) {
        value(c, returned)
        c.error(
            E0408_RETURN_TYPE_MISMATCH,
            node.span,
            "a `-> void` function must not return a value"
        )
        return
    }

    if want is not null { c.apply_context(c, returned, want) }
    let got = value(c, returned)
    if want is null { return }
    let _ok = c.require_assignable(
        c,
        returned,
        got,
        want,
        E0408_RETURN_TYPE_MISMATCH,
        "mismatched return type"
    )
}

fn _check_throw(c: Ctx, node: Node) -> void {
    let thrown = node.child("value")
    if thrown is null { return }
    let t = value(c, thrown)
    if is_any(t) or is_never(t) { return }
    if t.kind == TyKind.Class or t.kind == TyKind.Trait or t.kind == TyKind.GenericParam { return }

    let got = render(t)
    let d = c.error(
        E0400_TYPE_MISMATCH,
        node.span,
        f"only an `Error` instance can be thrown, found `{got}`"
    )
    d.with_help("throw a class deriving from `Error`, e.g. `ValueError(\"...\")`")
}

fn _check_try(c: Ctx, node: Node) -> void {
    let body = node.child("body")
    if body is not null { stmt(c, body) }

    for handler in node.records("catches") {
        let written: list[TypeExpr] = handler.get("types", [])
        var caught: Type? = null
        if written.len() > 0 {
            var members: list[Type?] = []
            for entry in written { members.push(c.resolve_ast_type(c, entry)) }
            if members.len() == 1 {
                caught = members[0]
            } else {
                caught = union_of(c.universe, members)
            }
        } else {
            # An unannotated `catch` binds `Error`. `Ctx.lookup_type_name`
            # answers null until the prelude registers it, so until then the
            # binding is `any`.
            caught = or_any(c, c.lookup_type_name(c, "Error"))
        }
        c.set_symbol_type(c.resolution.symbol_of(handler), or_any(c, caught))
        let handler_body: Node? = handler.get("body", null)
        if handler_body is not null { stmt(c, handler_body) }
    }

    let ensure = node.child("finallyBlock")
    if ensure is not null { stmt(c, ensure) }
}

fn _check_assert(c: Ctx, node: Node) -> void {
    condition(c, node.child("cond"), "an `assert`")
    let message = node.child("message")
    if message is null { return }
    let t = value(c, message)
    if is_any(t) or t.kind == TyKind.Str { return }
    let _ok = c.require_assignable(
        c,
        message,
        t,
        c.universe.t_str,
        E0400_TYPE_MISMATCH,
        "an assertion message must be a `str`"
    )
}

#: Check one statement, and everything under it.
#:
#: `jaiChkStmt`, and the shape of the installed `Ctx.check_stmt`. A declaration
#: goes to `Ctx.check_decl`, whose default does nothing.
pub fn stmt(c: Ctx, node: Node?) -> void {
    if node is null { return }

    match node.kind {
        NodeKind.Block | NodeKind.Program => { block(c, node) }

        NodeKind.ExprStmt => {
            let inner = node.child("expr")
            if inner is not null { expr(c, inner) }
        }

        NodeKind.VarDecl => { _check_var_decl(c, node) }
        NodeKind.Assign => { _check_assign(c, node) }
        NodeKind.If => { _check_if(c, node) }
        NodeKind.While | NodeKind.Loop => { _check_loop(c, node) }
        NodeKind.For => { _check_for(c, node) }
        NodeKind.Match => { c.set_type(node, match_expr(c, node, true)) }
        NodeKind.Return => { _check_return(c, node) }
        NodeKind.Throw => { _check_throw(c, node) }
        NodeKind.Try => { _check_try(c, node) }
        NodeKind.Assert => { _check_assert(c, node) }

        NodeKind.Defer => {
            let body = node.child("body")
            if body is not null { stmt(c, body) }
        }

        NodeKind.Break | NodeKind.Continue => { }

        _ => {
            if is_declaration(node.kind) {
                c.check_decl(c, node)
            } elif is_expression(node.kind) {
                expr(c, node)
            }
        }
    }
}
