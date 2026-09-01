#: A function as a signature and as a body, and the `install` that hands the
#: whole directory to `Ctx`.
#:
#: `function_type` reads a declaration and answers its `Fn` type; `function`
#: walks the body under that signature. Both push the generic parameters first,
#: and both restore what they pushed, so a nested function leaves nothing
#: behind: the generic names, the narrowing stack, the enclosing class and the
#: enclosing result type are all saved and put back.
#:
#: Two omissions mean `void` rather than `any`: a constructor, and a property
#: setter, which reaches here as a method named for the property. Everything
#: else with no written result is `any`, which is what makes an unannotated
#: function unchecked rather than wrong.
#:
#: `install` is last because it is the only place that names every entry point
#: at once. It fills in five `Ctx` callbacks and is the reason an expression can
#: reach a statement at all. `checker.jai` calls it and then replaces
#: `Ctx.check_decl`, `Ctx.resolve_ast_type`, `Ctx.layout_decl`, `Ctx.is_module`
#: and `Ctx.module_member`, none of which this directory can supply. Until it
#: does, every written annotation reads as `any`, a declaration inside a block
#: is unchecked, and a class is never laid out.
from jaithon.ast import Node, NodeKind, Span, TypeExpr, TypeKind, always_diverges
from jaithon.compile.check.ctx import Ctx, decl_type, is_any, is_never, is_void, or_any
from jaithon.compile.check.expr import E0400_TYPE_MISMATCH, self_skip_of, value
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.predicate import is_optional
from jaithon.compile.check.render import render
from jaithon.compile.check.stmt.codes import E0408_RETURN_TYPE_MISMATCH, E0601_MISSING_RETURN
from jaithon.compile.check.stmt.flow import block, stmt
from jaithon.compile.check.stmt.matching import match_expr
from jaithon.compile.check.stmt.pattern import pattern
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.universe import dict_of, fn_of, generic_param, list_of

#: Bind each generic parameter as a type name for the declaration that wrote it.
#:
#: Bounds are resolved after the parameters are in scope, so `[T: Comparable[T]]`
#: sees itself. The caller takes a `Ctx.type_name_mark` first and restores to it
#: when the declaration ends.
pub fn push_generics(c: Ctx, generics: list[dict[str, any]]) -> void {
    for entry in generics {
        let name: str? = entry.get("name", null)
        c.push_type_name(name, generic_param(c.universe, name))
    }
    for entry in generics {
        let bound: TypeExpr? = entry.get("bound", null)
        if bound is not null { c.resolve_ast_type(c, bound) }
    }
}

#: Whether `fn_node` is a constructor.
pub fn is_init_method(fn_node: Node?) -> bool {
    if fn_node is null or fn_node.kind != NodeKind.FnDecl { return false }
    return fn_node.opt_text("name") == "init"
}

#: The declared return type, with the two places omission means `void`.
#:
#: Everything else with no written result is `any`; a constructor is `void`, and
#: so is a property setter, which reaches here as a method named for the
#: property.
pub fn declared_return_type(c: Ctx, fn_node: Node?, is_method: bool) -> Type {
    let u = c.universe
    if fn_node is null { return u.t_any }
    let written = fn_node.type_field("returnType")
    if written is not null and written.kind != TypeKind.Infer {
        return or_any(c, c.resolve_ast_type(c, written))
    }
    if is_method and is_init_method(fn_node) { return u.t_void }
    return u.t_any
}

#: The `Fn` type of a declaration, lambda or anonymous function.
#:
#: A method's receiver is dropped, because a call site never passes it. The
#: generic parameters are in scope only while the parameter and result types are
#: resolved, which is what makes `fn map[T](xs: list[T]) -> list[T]` resolve `T`.
pub fn function_type(c: Ctx, fn_node: Node?, is_method: bool) -> Type {
    let u = c.universe
    if fn_node is null { return u.t_any }

    let mark = c.type_name_mark()
    push_generics(c, fn_node.records("generics"))

    let all_params = fn_node.records("params")
    let skip = if is_method { self_skip_of(fn_node) } else { 0 }
    var count = all_params.len() - skip
    if count < 0 { count = 0 }

    var flags = 0
    var params: list[Type] = []
    var at = 0
    while at < count {
        let p = all_params[skip + at]
        if p.get("isVariadic", false) { flags = flags | 1 }
        if p.get("isKwRest", false) { flags = flags | 2 }
        let written: TypeExpr? = p.get("type", null)
        var declared: Type? = null
        if written is not null { declared = c.resolve_ast_type(c, written) }
        params.push(or_any(c, declared))
        at += 1
    }

    let returns = declared_return_type(c, fn_node, is_method)
    c.type_name_restore(mark)
    return fn_of(u, params, returns, flags)
}

# The type a parameter's binding has inside the body, which is not the type in
# the signature for `...xs` and `**opts`.
fn _parameter_binding_type(c: Ctx, p: dict[str, any]) -> Type {
    let u = c.universe
    let written: TypeExpr? = p.get("type", null)
    var declared: Type? = null
    if written is not null { declared = c.resolve_ast_type(c, written) }
    let settled = or_any(c, declared)
    if p.get("isVariadic", false) { return list_of(u, settled) }
    if p.get("isKwRest", false) { return dict_of(u, u.t_str, settled) }
    return settled
}

#: Check a function body against the signature it was written with.
#:
#: `jaiChkFunction`. The generic names, the narrowing stack, the enclosing class
#: and the enclosing result type are all saved and restored, so a nested function
#: leaves nothing behind. `owner` is the class whose body this method is in, and
#: `is_method` is what makes the first parameter a receiver.
#:
#: E0601 — an annotated non-`void` function must not fall off the end — is the
#: one diagnostic here about the body as a whole. A generator returns through
#: `yield`, so it is exempt.
pub fn function(c: Ctx, fn_node: Node?, owner: TypeDecl?, is_method: bool) -> void {
    if fn_node is null { return }
    let u = c.universe

    let name_mark = c.type_name_mark()
    push_generics(c, fn_node.records("generics"))

    let narrow_mark = c.narrow_mark()
    let saved_return = c.current_return_type
    let saved_class = c.current_class
    if owner is not null { c.current_class = owner }

    let skip = if is_method { self_skip_of(fn_node) } else { 0 }
    for (position, p) in fn_node.records("params").enumerate() {
        var bound: Type = u.t_any
        if position < skip {
            bound = decl_type(c, owner)
        } else {
            bound = _parameter_binding_type(c, p)
            let written: TypeExpr? = p.get("type", null)
            if c.strict and written is null and not p.get("isKwRest", false) {
                let shown: str? = p.get("name", null)
                let span: Span = p.get("span", Span.none())
                let d = c.error(
                    E0400_TYPE_MISMATCH,
                    span,
                    f"parameter `{shown ?? "?"}` has no type annotation"
                )
                d.with_help("--strict requires every parameter to be annotated")
            }
        }
        c.set_symbol_type(c.resolution.symbol_of(p), bound)
        let fallback: Node? = p.get("defaultValue", null)
        if fallback is not null {
            c.apply_context(c, fallback, bound)
            let got = value(c, fallback)
            let _ok = c.require_assignable(
                c,
                fallback,
                got,
                bound,
                E0400_TYPE_MISMATCH,
                "mismatched type for a default value"
            )
        }
    }

    let written_return = fn_node.type_field("returnType")
    let annotated = written_return is not null and written_return.kind != TypeKind.Infer
    var returns = declared_return_type(c, fn_node, is_method)
    c.current_return_type = returns

    let body = fn_node.child("body")
    if body is not null {
        if body.kind == NodeKind.Block {
            block(c, body)
        } else {
            # A lambda's body is its return value.
            if not is_any(returns) { c.apply_context(c, body, returns) }
            let got = value(c, body)
            if annotated {
                let _ok = c.require_assignable(
                    c,
                    body,
                    got,
                    returns,
                    E0408_RETURN_TYPE_MISMATCH,
                    "mismatched return type"
                )
            } else {
                returns = got
            }
        }
    }

    if body is not null and
        body.kind == NodeKind.Block and
        not fn_node.flag("isGenerator") and
        annotated and
        not is_void(c, returns) and
        not is_any(returns) and
        not is_never(returns) and
        returns.kind != TyKind.Null and
        not is_optional(returns) and
        not always_diverges(body) {
        let want = render(returns)
        let shown = fn_node.opt_text("name") ?? "this function"
        let d = c.error(
            E0601_MISSING_RETURN,
            fn_node.span,
            f"`{shown}` must return `{want}` on every path"
        )
        if written_return is not null {
            d.label(written_return.span, f"declared to return `{want}`")
        }
        d.with_help("add a `return` at the end, or make the last branch an `else`")
    }

    c.current_return_type = saved_return
    c.current_class = saved_class
    c.narrow_restore(narrow_mark)
    c.type_name_restore(name_mark)
}

#: Install this file's entries into `c`, so an expression can reach a statement.
#:
#: Without this call a comprehension's loop variable stays unbound, a `match` in
#: expression position answers `any`, a block or statement in expression position
#: is not walked, and a lambda body is not checked — each silently.
#:
#: `checker.jai` will call this and then install `Ctx.check_decl`,
#: `Ctx.resolve_ast_type`, `Ctx.layout_decl`, `Ctx.is_module` and
#: `Ctx.module_member`, none of which this file can supply.
pub fn install(c: Ctx) -> void {
    c.check_pattern = pattern
    c.check_block = block
    c.check_stmt = stmt
    c.match_expr = match_expr
    c.check_function = _check_lambda
    c.function_type = _lambda_type
}

fn _check_lambda(c: Ctx, fn_node: Node?) -> void {
    function(c, fn_node, null, false)
}

fn _lambda_type(c: Ctx, fn_node: Node?) -> Type {
    return function_type(c, fn_node, false)
}
