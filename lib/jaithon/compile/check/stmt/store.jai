#: The two ways a body puts a value somewhere: a `let`/`var`/`const` that binds
#: a new name, and an assignment that overwrites an existing place.
#:
#: A store is checked against the **declaration**, not against whatever the
#: narrowing stack currently believes — `Ctx.declared_type` — and the caller in
#: `flow.jai` then ends the fact with `Ctx.narrow_invalidate`. Widening a store
#: to the narrowed type instead would let `x` hold a value its declaration
#: forbids.
#:
#: Five places can be assigned and each fails differently: a name (immutability
#: is the resolver's business, so only the type is checked here), a field
#: (visibility, `let`, a setter, a method that is not a field, a name the class
#: never declared), an index (the container's key and element types), a slice,
#: and a destructuring pattern. A compound `x += y` is checked as the operator
#: rather than as a plain store, so `int += float` gets `E0411` and its own
#: help rather than a bare mismatch.
#:
#: `const` folding runs here too. The folded value is written back onto the
#: initialiser and recorded in `Ctx.consts` **by name**, which is a known gap:
#: two `const K` in different scopes collide.
from jaithon.ast import DeclKind, Node, NodeKind, TypeKind, is_pattern, op_text
from jaithon.compile.check.ctx import Ctx, E0411_INT_FLOAT_MIX, is_any, is_never
from jaithon.compile.check.ctx import or_any, subject_of, type_decl_of
from jaithon.compile.check.expr import E0400_TYPE_MISMATCH, E0404_NOT_INDEXABLE
from jaithon.compile.check.expr import E0406_BAD_OPERAND_TYPES, E0701_PRIVATE_ACCESS
from jaithon.compile.check.expr import expr, slice, static_target_decl, value
from jaithon.compile.check.expr import visibility_word, visible_from
from jaithon.compile.check.fold import ConstKind, const_eval, const_replace
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.nominal import find_accessor, find_member, owner_of_field
from jaithon.compile.check.operator import binary_result, widens_operands
from jaithon.compile.check.predicate import is_indexable
from jaithon.compile.check.render import render
from jaithon.compile.check.stmt.codes import E0301_ASSIGN_TO_IMMUTABLE, E0303_NON_CONSTANT_CONST
from jaithon.compile.check.stmt.codes import E0304_MISSING_INITIALIZER, E0702_UNDECLARED_FIELD
from jaithon.compile.check.stmt.fields import _field_index, _layout_fields
from jaithon.compile.check.stmt.pattern import pattern
from jaithon.compile.check.suggest import suggest_member
from jaithon.compile.check.ty import Type
from jaithon.compile.symbol import Symbol

fn _decl_symbol(c: Ctx, node: Node) -> Symbol? {
    let own = c.resolution.symbol_of(node)
    if own is not null { return own }
    let pat = node.child("pattern")
    if pat is not null and pat.kind == NodeKind.PatBind { return c.resolution.symbol_of(pat) }
    return null
}

fn _widen_inferred(c: Ctx, kind: DeclKind, t: Type?) -> Type {
    let u = c.universe
    if t is null { return u.t_any }
    if kind == DeclKind.Var and (t.kind == TyKind.Null or t.kind == TyKind.Never) {
        return u.t_any
    }
    if t.kind == TyKind.Never { return u.t_any }
    return t
}

fn _fold_const_decl(c: Ctx, node: Node, symbol: Symbol?) -> void {
    let init = node.child("init")
    if init is null {
        c.error(E0304_MISSING_INITIALIZER, node.span, "a `const` declaration needs an initialiser")
        return
    }

    let before = c.bag.error_count
    let v = const_eval(init, c.bag, c.consts)

    if v.kind == ConstKind.None {
        if c.bag.error_count == before {
            let d = c.error(
                E0303_NON_CONSTANT_CONST,
                init.span,
                "the initialiser of a `const` must be a constant expression"
            )
            d.with_help("use `let` if the value is only known at run time")
        }
        return
    }

    let literal = Node.null_lit().with_span(init.span)
    c.set_type(literal, const_replace(c.universe, literal, v))
    c.set_type(init, const_replace(c.universe, init, v))
    if symbol is not null { c.consts[symbol.name] = literal }
}

pub fn _check_var_decl(c: Ctx, node: Node) -> void {
    let annotation = node.type_field("declaredType")
    var declared: Type? = null
    if annotation is not null and annotation.kind != TypeKind.Infer {
        declared = c.resolve_ast_type(c, annotation)
    }
    let init = node.child("init")
    var bound = declared

    if init is not null {
        if declared is not null { c.apply_context(c, init, declared) }
        let got = value(c, init)
        if declared is not null {
            let _ok = c.require_assignable(
                c,
                init,
                got,
                declared,
                E0400_TYPE_MISMATCH,
                "mismatched types in this binding"
            )
        } else {
            bound = _widen_inferred(c, node.get("varDeclKind"), got)
        }
    }

    let settled = or_any(c, bound)
    c.set_type(node, settled)

    let symbol = _decl_symbol(c, node)
    c.set_symbol_type(symbol, settled)
    let pat = node.child("pattern")
    if pat is not null { pattern(c, pat, settled) }

    let kind: DeclKind = node.get("varDeclKind")
    if kind == DeclKind.Const { _fold_const_decl(c, node, symbol) }
}

fn _check_stored_value(c: Ctx, source: Node?, want: Type?) -> void {
    if want is not null and not is_any(want) { c.apply_context(c, source, want) }
    let got = value(c, source)
    if want is null { return }
    let _ok = c.require_assignable(
        c,
        source,
        got,
        want,
        E0400_TYPE_MISMATCH,
        "mismatched types in this assignment"
    )
}

fn _check_field_assign(c: Ctx, node: Node, target: Node) -> void {
    let u = c.universe
    let object = target.child("object")
    let source = node.child("value")
    let name = target.opt_text("name") ?? ""

    var found = static_target_decl(c, object)
    let static_access = found is not null
    if found is null { found = type_decl_of(value(c, object)) }
    if found is null {
        c.set_type(target, u.t_any)
        value(c, source)
        return
    }
    let decl = found
    c.layout_decl(c, decl)

    let fields = _layout_fields(decl)
    let fi = _field_index(decl, name)
    if fi >= 0 {
        let field = fields[fi]
        let owner = owner_of_field(decl, name)
        if not visible_from(c, owner, field.visibility) {
            let word = visibility_word(field.visibility)
            let owner_name = if owner is null { decl.name } else { owner.name }
            let d = c.error(
                E0701_PRIVATE_ACCESS,
                target.span,
                f"`{name}` is {word} to `{owner_name}`"
            )
            if owner is not null { d.label(owner.span, "declared here") }
        }
        if field.is_let and not static_access and c.current_class != decl {
            let d = c.error(
                E0301_ASSIGN_TO_IMMUTABLE,
                target.span,
                f"`{name}` is declared `let` and cannot be reassigned"
            )
            d.with_help("declare it `var` if it needs to change after `init`")
        }

        c.set_type(target, or_any(c, field.ty))
        _check_stored_value(c, source, c.type_of(target))
        return
    }

    let setter = find_accessor(decl, name, true)
    if setter is not null {
        c.set_type(target, or_any(c, setter.ty))
        _check_stored_value(c, source, c.type_of(target))
        return
    }

    c.set_type(target, u.t_any)
    if find_member(decl, name) is not null {
        c.error(
            E0702_UNDECLARED_FIELD,
            target.span,
            f"`{name}` is a method of `{decl.name}` and cannot be assigned"
        )
    } else {
        let d = c.error(E0702_UNDECLARED_FIELD, target.span, f"`{decl.name}` has no field `{name}`")
        d.label(decl.span, f"`{decl.name}` is declared here")
        suggest_member(d, decl, name)
        d.with_help("declare it in the class body; fields are not created by assignment")
    }
    value(c, source)
}

fn _check_compound_assign(c: Ctx, node: Node, target: Node, source: Node) -> void {
    let u = c.universe
    let current = expr(c, target)
    var rhs = value(c, source)
    if is_any(current) or is_any(rhs) or is_never(current) or is_never(rhs) { return }

    let op = node.op()
    if widens_operands(op) and current.kind == TyKind.Float and rhs.kind == TyKind.Int {
        c.widen_to_float(c, source, u.t_float)
        rhs = u.t_float
    }

    if widens_operands(op) and current.kind == TyKind.Int and rhs.kind == TyKind.Float {
        let word = op_text(op)
        let d = c.error(
            E0411_INT_FLOAT_MIX,
            node.span,
            f"cannot apply `{word}=` to `int` and `float`"
        )
        let subject = subject_of(target, "the target")
        d.with_help(
            f"the result is `float` and `{subject}` is `int`; " +
                "declare it `float`, or truncate with `int(...)`"
        )
        return
    }

    let result = binary_result(u, op, current, rhs)
    if result is null {
        let word = op_text(op)
        let a = render(current)
        let b = render(rhs)
        c.error(
            E0406_BAD_OPERAND_TYPES,
            node.span,
            f"operator `{word}=` does not apply to `{a}` and `{b}`"
        )
        return
    }
    let _ok = c.require_assignable(
        c,
        null,
        result,
        current,
        E0400_TYPE_MISMATCH,
        "the result of this compound assignment does not fit"
    )
}

fn _check_index_assign(c: Ctx, node: Node, target: Node) -> void {
    let u = c.universe
    let source = node.child("value")
    let container = value(c, target.child("object"))
    let index_node = target.child("index")
    let index = value(c, index_node)

    var element: Type? = null
    let shape = if is_any(container) { null } else { is_indexable(u, container) }
    if shape is not null {
        let (want_index, want_element) = shape
        if not is_any(index) {
            let _ok = c.require_assignable(
                c,
                index_node,
                index,
                want_index,
                E0400_TYPE_MISMATCH,
                "invalid index type"
            )
        }
        element = want_element
        c.apply_context(c, source, want_element)
    } elif not is_any(container) and type_decl_of(container) is null {
        let got = render(container)
        c.error(E0404_NOT_INDEXABLE, target.span, f"`{got}` does not support indexed assignment")
    }

    let got = value(c, source)
    if element is not null {
        let _ok = c.require_assignable(
            c,
            source,
            got,
            element,
            E0400_TYPE_MISMATCH,
            "mismatched types in this assignment"
        )
    }
    c.set_type(target, or_any(c, element))
}

pub fn _check_assign(c: Ctx, node: Node) -> void {
    let target = node.child("target")
    let source = node.child("value")
    if target is null or source is null { return }

    if node.flag("isCompound") {
        _check_compound_assign(c, node, target, source)
        return
    }

    match target.kind {
        NodeKind.Ident => {
            var want = expr(c, target)
            let symbol = c.resolution.symbol_of(target)
            if symbol is not null and c.symbol_type(symbol) is not null {
                want = or_any(c, c.declared_type(symbol))
            }
            c.set_type(target, or_any(c, want))
            _check_stored_value(c, source, want)
            c.narrow_invalidate(symbol)
        }

        NodeKind.Member | NodeKind.OptMember => { _check_field_assign(c, node, target) }

        NodeKind.Index => { _check_index_assign(c, node, target) }

        NodeKind.Slice => {
            slice(c, target)
            value(c, source)
        }

        _ => {
            if is_pattern(target.kind) {
                let got = value(c, source)
                pattern(c, target, got)
            } else {
                expr(c, target)
                value(c, source)
            }
        }
    }
}
