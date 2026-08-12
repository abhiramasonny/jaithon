#: The type of every expression form: operators, member access, calls,
#: indexing and slicing, comprehensions and function expressions.
#:
#: A port of `src/sema/check_expr.c`. Types live in side tables keyed by
#: `id(node)`, and everything the other checker files own arrives through `Ctx`
#: as a callback whose default is documented below.
#:
#: What is NOT here: the pure, `Ctx`-free material that used to live in this
#: file has moved out, each to the sibling it fits rather than to a new file
#: invented just to hold it. `is_arithmetic` and `widens_operands` — OpKind
#: classification with no operand types in sight — joined `binary_result` and
#: `unary_result` in `operator.jai`, which already tables "what an operator
#: does". The plain `dict[str, any]` field accessors for argument, parameter
#: and comprehension-clause records joined `arg_value`/`param_type`/
#: `param_variadic`/`param_kwrest` in `ctx.jai`, which already held that exact
#: kind of accessor. The one genuinely new file is `suggest.jai`: the bounded
#: edit-distance search and `suggest_member` did not fit any existing file's
#: charter, and are shared as-is with `stmt.jai`'s own "did you mean"s.
#: Everything that stayed is the dispatch in `expr` and the per-node-kind
#: checks it calls, which constantly recurse through `value`/`expr`/`member`
#: and thread `Ctx` for narrowing and diagnostics — one checking pass, not
#: independent pieces.
from jaithon.ast import CompKind, DeclKind, Node, NodeKind, OpKind, Span
from jaithon.ast import is_statement, op_is_comparison, op_text
from jaithon.compile.check.assign import widen_target
from jaithon.compile.check.ctx import Ctx, arg_name, arg_span, arg_spread, arg_value
from jaithon.compile.check.ctx import clause_conditions, clause_iterable, clause_pattern
from jaithon.compile.check.ctx import decl_type, is_any, is_never, is_type_symbol, is_void
from jaithon.compile.check.ctx import param_default, param_kwrest, param_name, param_type
from jaithon.compile.check.ctx import param_variadic, span_of, subject_of, type_decl_of
from jaithon.compile.check.fold import ConstKind, const_eval, const_replace, try_fold
from jaithon.compile.check.kinds import MAX_DECL_DEPTH, TyKind
from jaithon.compile.check.nominal import find_accessor, find_field, find_member
from jaithon.compile.check.nominal import inherits_opaquely, is_subclass_of
from jaithon.compile.check.nominal import names_opaque_trait, owner_of_field, owner_of_member
from jaithon.compile.check.operator import binary_result, unary_result, widens_operands
from jaithon.compile.check.predicate import is_callable, is_indexable, is_iterable, is_optional
from jaithon.compile.check.relate import join, narrow
from jaithon.compile.check.render import render
from jaithon.compile.check.substitute import mentions_generic_param, substitute
from jaithon.compile.check.suggest import SUGGEST_NO_MATCH, closer, suggest_member
from jaithon.compile.check.ty import MemberInfo, Type, TypeDecl
from jaithon.compile.check.union import optional_of
from jaithon.compile.check.universe import dict_of, list_of, set_of, tuple_of
from jaithon.compile.symbol import Symbol, SymbolKind

#: A value does not fit the slot it is going into.
#:
#: These are the C's `JaiDiagCode` names in `src/common/diag.h`. Both the code
#: and the message text are a contract: `tests/errors` turns them into goldens.
pub const E0400_TYPE_MISMATCH: str = "E0400"

#: A condition is not a `bool`. There is no truthiness.
pub const E0401_CONDITION_NOT_BOOL: str = "E0401"

#: A value that cannot appear before a `(`.
pub const E0403_NOT_CALLABLE: str = "E0403"

#: A value that cannot be subscripted or sliced.
pub const E0404_NOT_INDEXABLE: str = "E0404"

#: A value that cannot be looped over.
pub const E0405_NOT_ITERABLE: str = "E0405"

#: An operator that does not apply to the operands it was given.
pub const E0406_BAD_OPERAND_TYPES: str = "E0406"

#: A `void` expression used where a value was wanted.
pub const E0409_VOID_VALUE_USED: str = "E0409"

#: A member that the declaration does not have.
pub const E0410_UNKNOWN_MEMBER: str = "E0410"

#: A call of an unnamed callable with the wrong number of arguments.
pub const E0600_ARITY_MISMATCH: str = "E0600"

#: A call of a named function with the wrong number of arguments.
pub const E0602_BAD_CALL_ARITY: str = "E0602"

#: A keyword argument naming no parameter.
pub const E0603_UNKNOWN_KEYWORD_ARG: str = "E0603"

#: One argument given twice.
pub const E0604_DUPLICATE_ARGUMENT: str = "E0604"

#: A positional argument after a keyword argument.
pub const E0605_POSITIONAL_AFTER_KEYWORD: str = "E0605"

#: A member reached from outside the visibility it was declared with.
pub const E0701_PRIVATE_ACCESS: str = "E0701"

#: A trait or an abstract class used as a constructor.
pub const E0710_ABSTRACT_INSTANTIATION: str = "E0710"

#: Code that cannot run — here, the right operand of a pointless `??`.
pub const W0102_UNREACHABLE_CODE: str = "W0102"

#: A member or field declared with no modifier.
#:
#: `ty.jai` holds visibility as a plain `int`; these three are the ordinals of
#: `jaithon.ast.Visibility`, which is the order of the C `AstVisibility`.
pub const VIS_PRIVATE: int = 0

#: Reachable from the declaring class and its subclasses.
pub const VIS_PROTECTED: int = 1

#: Reachable from anywhere: the declaration was marked `pub`.
pub const VIS_PUBLIC: int = 2

#: How many arguments or parameters one call binds, matching `JAI_MAX_ARGS`.
#:
#: Reproduced because it is observable: past it the C stops binding arguments to
#: parameters, so a port that kept binding would report errors the C does not.
pub const MAX_ARGS: int = 255

# The type an already-checked element contributes to a container's join. `value`
# reports `any` for a void expression but records `void`, so re-reading the tree
# has to normalise the same way.
fn _element_type(c: Ctx, node: Node?) -> Type {
    if node is null { return c.universe.t_any }
    let t = c.type_of(node)
    if is_void(c, t) { return c.universe.t_any }
    return t ?? c.universe.t_any
}

# The contextual type `apply_context` stamped on a container literal, if it is
# the right shape. Consumed, so a stale hint cannot leak into a later literal.
fn _take_container_context(c: Ctx, node: Node, want: TyKind, arity: int) -> Type? {
    let hint = c.type_of(node)
    c.clear_type(node)
    if hint is null or hint.kind != want or hint.arg_count() != arity { return null }
    return hint
}

#: Check that `node` is a `bool`, naming `where` in the diagnostic. Every boolean
#: position in the language funnels through here.
pub fn condition(c: Ctx, node: Node?, where: str) -> void {
    if node is null { return }
    let t = value(c, node)
    if is_any(t) or is_never(t) or t.kind == TyKind.Bool { return }

    let got = render(t)
    let d = c.error(
        E0401_CONDITION_NOT_BOOL,
        node.span,
        f"the condition of {where} must be `bool`, found `{got}`"
    )
    let help = _condition_help(t, node)
    if help is not null { d.with_help(help) }
}

# Written with the condition's own spelling, so the help is a line to paste.
fn _condition_help(t: Type?, node: Node?) -> str? {
    if t is null { return null }
    if is_optional(t) or t.kind == TyKind.Null {
        let subject = subject_of(node, "x")
        return f"test for absence explicitly: `{subject} != null`"
    }
    match t.kind {
        TyKind.List | TyKind.Dict | TyKind.Set | TyKind.Str | TyKind.Bytes | TyKind.Tuple => {
            let subject = subject_of(node, "xs")
            return f"test the length explicitly: `{subject}.len() > 0`"
        }
        TyKind.Int | TyKind.Float => {
            let subject = subject_of(node, "n")
            return f"compare explicitly: `{subject} != 0`"
        }
        _ => { return "Jaithon has no truthiness; a condition must be a `bool`" }
    }
}

#: What `for x in node` binds, reporting E0405 when `t` cannot be looped over.
#: A user type is iterable when it implements `Iterable` (spec §5.2), which the
#: type system cannot see through a bare class type, so the declaration is asked.
pub fn iterable_element(c: Ctx, node: Node?, t: Type?) -> Type {
    let u = c.universe
    if is_any(t) or is_never(t) { return u.t_any }

    # Lay the declaration out first: an imported one that has not been would
    # answer "no members, no traits" and be reported as not iterable.
    let iter_decl = type_decl_of(t)
    if iter_decl is not null { c.layout_decl(c, iter_decl) }

    let elem = is_iterable(u, t)
    if elem is not null { return elem }

    if t is not null and (t.kind == TyKind.Class or t.kind == TyKind.Trait) {
        if _class_is_iterable(type_decl_of(t)) { return u.t_any }
    }

    let got = render(t)
    let d = c.error(E0405_NOT_ITERABLE, span_of(node), f"`{got}` is not iterable")
    if t is not null and t.kind == TyKind.Class {
        d.with_help(f"implement `trait Iterable` on `{got}` to loop over it")
    } elif t is not null and t.kind == TyKind.Int {
        d.with_help("loop over a range: `for i in 0..n`")
    }
    return u.t_any
}

# An unknown shape is assumed iterable. `opaque_traits` is consulted alongside
# `traits` because a class implementing an imported `Iterable` names it there.
fn _class_is_iterable(decl: TypeDecl?) -> bool {
    if decl is null { return true }
    if find_member(decl, "__iter__") is not null { return true }
    if find_member(decl, "__next__") is not null { return true }
    for implemented in decl.traits {
        if implemented.name == "Iterable" { return true }
    }
    return names_opaque_trait(decl, "Iterable")
}

#: The type of `node` used as a value, rather than for its effect. `expr` plus
#: the rule that a `void` expression produces no value, which reports nothing
#: today: `void` and `null` are one interned type. See `is_void`.
pub fn value(c: Ctx, node: Node?) -> Type {
    let t = expr(c, node)
    if is_void(c, t) {
        c.error(
            E0409_VOID_VALUE_USED,
            span_of(node),
            "this expression has type `void` and produces no value"
        )
        return c.universe.t_any
    }
    return t
}

fn _check_ident(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let symbol = c.resolution.symbol_of(node)
    # The resolver already reported the undefined name; do not pile on.
    if symbol is null { return u.t_any }

    # A folded `const` is inlined at its use site (spec §3).
    if c.fold_constants and symbol.mutability == DeclKind.Const and symbol.name in c.consts {
        let folded = const_eval(node, null, c.consts)
        if not folded.is_none() {
            let literal = const_replace(u, node, folded)
            c.set_type(node, c.symbol_type(symbol) ?? literal)
            return c.type_of(node) ?? u.t_any
        }
    }

    # Classes and enums may be referenced before their layout pass ran. The test
    # is on the resolved type, not only the symbol kind: this front end has no
    # runtime symbol registration, so `ValueError` would answer `any`. Locals and
    # parameters stay excluded, so a binding that shadows a class name wins.
    if c.symbol_type(symbol) is null and symbol.kind != SymbolKind.Local {
        if symbol.kind != SymbolKind.Param {
            let named = c.lookup_type_name(c, symbol.name)
            let nominal = named is not null and named.is_named()
            if named is not null and (is_type_symbol(symbol.kind) or nominal) {
                c.set_symbol_type(symbol, named)
            }
        }
    }
    return c.symbol_type(symbol) ?? u.t_any
}

# Numeric fusion (spec §2.2): when an unannotated literal position mixes an exact
# `int` with an exact `float`, every int node is converted in place. Nothing is
# returned — the caller joins by re-reading the recorded types, so a node that
# failed to convert leaves the join a union rather than a lie.
fn _fuse_numeric(c: Ctx, nodes: list[Node]) -> void {
    let u = c.universe
    var has_int = false
    var has_float = false
    for node in nodes {
        let t = c.type_of(node)
        # `widen_target` is the exact-int test: it resolves to null for `int?`,
        # `any`, `never` and `list[int]` alike.
        if widen_target(u, t, u.t_float) is not null { has_int = true }
        if t == u.t_float { has_float = true }
    }
    if not has_int or not has_float { return }

    for node in nodes {
        if widen_target(u, c.type_of(node), u.t_float) is null { continue }
        c.widen_to_float(c, node, u.t_float)
    }
}

fn _check_sequence_literal(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let items = node.child_list("items")

    if node.kind == NodeKind.TupleLit {
        let want_tuple = _take_container_context(c, node, TyKind.Tuple, items.len())
        if want_tuple is not null {
            for (position, item) in items.enumerate() {
                let got = value(c, item)
                let _ok = c.require_assignable(
                    c,
                    item,
                    got,
                    want_tuple.args[position],
                    E0400_TYPE_MISMATCH,
                    "mismatched member type"
                )
            }
            return want_tuple
        }
        for item in items { value(c, item) }
        var members: list[Type] = []
        for item in items { members.push(_element_type(c, item)) }
        return tuple_of(u, members)
    }

    let shape = if node.kind == NodeKind.SetLit { TyKind.Set } else { TyKind.List }
    let want = _take_container_context(c, node, shape, 1)

    # Annotated: each element is checked against the declared element type.
    if want is not null {
        for item in items {
            let got = value(c, item)
            let _ok = c.require_assignable(
                c,
                item,
                got,
                want.args[0],
                E0400_TYPE_MISMATCH,
                "mismatched element type"
            )
        }
        return want
    }

    # Unannotated: check, fuse, then join by re-reading the tree. Folding the join
    # as we went would make the result order-dependent — `[1, "a", 2.0]` would
    # never see the int and the float as a pair.
    #
    # A set literal is deliberately not fused: a number in a hash position decides
    # which elements are the same element, so converting one can merge two members
    # that were distinct above 2^53.
    for item in items { value(c, item) }
    if node.kind == NodeKind.ListLit { _fuse_numeric(c, items) }

    var elem: Type? = null
    for item in items {
        let t = _element_type(c, item)
        if elem is null { elem = t } else { elem = join(u, elem, t) }
    }
    let element = elem ?? u.t_any
    return if node.kind == NodeKind.SetLit { set_of(u, element) } else { list_of(u, element) }
}

fn _check_dict_literal(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let keys = node.child_list("keys")
    let values = node.child_list("values")
    let want = _take_container_context(c, node, TyKind.Dict, 2)

    if want is not null {
        for (position, key) in keys.enumerate() {
            let got = value(c, key)
            let _ok = c.require_assignable(
                c,
                key,
                got,
                want.args[0],
                E0400_TYPE_MISMATCH,
                "mismatched key type"
            )
            if position >= values.len() { continue }
            let held = values[position]
            let held_type = value(c, held)
            let _fits = c.require_assignable(
                c,
                held,
                held_type,
                want.args[1],
                E0400_TYPE_MISMATCH,
                "mismatched value type"
            )
        }
        return want
    }

    # Unannotated: check, fuse the values, then join by re-reading the tree — see
    # `_check_sequence_literal`. Keys are a hash position and are left alone. Key
    # then value, entry by entry, is the order the C reports two errors in.
    for (position, key) in keys.enumerate() {
        value(c, key)
        if position < values.len() { value(c, values[position]) }
    }
    _fuse_numeric(c, values)

    var key_type: Type? = null
    var value_type: Type? = null
    for key in keys {
        let k = _element_type(c, key)
        if key_type is null { key_type = k } else { key_type = join(u, key_type, k) }
    }
    for held in values {
        let v = _element_type(c, held)
        if value_type is null { value_type = v } else { value_type = join(u, value_type, v) }
    }
    return dict_of(u, key_type ?? u.t_any, value_type ?? u.t_any)
}

# One int, one float: the int side widens and the expression is float, with the
# conversion in the tree rather than left to the VM.
fn _widen_numeric_mix(c: Ctx, node: Node, op: OpKind, left: Type, right: Type) -> (Type, Type) {
    let u = c.universe
    if not widens_operands(op) { return (left, right) }
    let left_int = left.kind == TyKind.Int and right.kind == TyKind.Float
    let right_int = right.kind == TyKind.Int and left.kind == TyKind.Float
    if not left_int and not right_int { return (left, right) }

    let int_side = if left_int { node.child("left") } else { node.child("right") }
    c.widen_to_float(c, int_side, u.t_float)
    return if left_int { (u.t_float, right) } else { (left, u.t_float) }
}

fn _check_binary(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let op = node.op()
    let checked_left = value(c, node.child("left"))
    let checked_right = value(c, node.child("right"))

    # Identity never constrains its operands: `x is Circle`, or `x is y`.
    if op == OpKind.Is or op == OpKind.IsNot { return u.t_bool }

    let (left, right) = _widen_numeric_mix(c, node, op, checked_left, checked_right)
    if is_never(left) or is_never(right) { return u.t_never }
    if is_any(left) or is_any(right) {
        return if op_is_comparison(op) { u.t_bool } else { u.t_any }
    }

    let result = binary_result(u, op, left, right)
    if result is not null { return result }

    let a = render(left)
    let b = render(right)
    let d = c.error(
        E0406_BAD_OPERAND_TYPES,
        node.span,
        f"operator `{op_text(op)}` does not apply to `{a}` and `{b}`"
    )
    if type_decl_of(left) is not null {
        d.with_help(f"define the matching dunder method on `{a}`")
    }
    return u.t_any
}

fn _check_logical(c: Ctx, node: Node) -> Type {
    let op = node.op()
    let what = if op == OpKind.And { "`and`" } else { "`or`" }
    let left = node.child("left")
    condition(c, left, what)

    # The right operand of `and` sees the facts the left operand established.
    let mark = c.narrow_mark()
    let facts = c.collect_facts(left, op == OpKind.Or)
    c.narrow_apply(facts, false)
    condition(c, node.child("right"), what)
    c.narrow_restore(mark)
    return c.universe.t_bool
}

fn _check_chain(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let ops: list[OpKind] = node.get("ops")
    let operands = node.child_list("operands")
    if ops.len() <= 0 or operands.len() != ops.len() + 1 { return u.t_bool }

    for operand in operands { value(c, operand) }

    for (position, op) in ops.enumerate() {
        if op == OpKind.Is or op == OpKind.IsNot { continue }
        let left = c.type_of(operands[position]) ?? u.t_any
        let right = c.type_of(operands[position + 1]) ?? u.t_any
        if is_any(left) or is_any(right) or is_never(left) or is_never(right) { continue }
        if binary_result(u, op, left, right) is not null { continue }

        let a = render(left)
        let b = render(right)
        c.error(
            E0406_BAD_OPERAND_TYPES,
            node.span,
            f"operator `{op_text(op)}` does not apply to `{a}` and `{b}`"
        )
    }
    return u.t_bool
}

fn _check_unary(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let op = node.op()
    if op == OpKind.Not {
        condition(c, node.child("operand"), "`not`")
        return u.t_bool
    }

    let operand = value(c, node.child("operand"))
    if is_any(operand) or is_never(operand) { return operand }

    let result = unary_result(u, op, operand)
    if result is not null { return result }

    let got = render(operand)
    c.error(
        E0406_BAD_OPERAND_TYPES,
        node.span,
        f"operator `{op_text(op)}` does not apply to `{got}`"
    )
    return u.t_any
}

fn _check_ternary(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let cond = node.child("cond")
    condition(c, cond, "a ternary")

    let facts = c.collect_facts(cond, false)
    var mark = c.narrow_mark()
    c.narrow_apply(facts, false)
    let then_type = value(c, node.child("thenExpr"))
    c.narrow_restore(mark)

    mark = c.narrow_mark()
    c.narrow_apply(facts, true)
    let else_type = value(c, node.child("elseExpr"))
    c.narrow_restore(mark)

    return join(u, then_type, else_type)
}

fn _check_coalesce(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let left_node = node.child("left")
    let right_node = node.child("right")
    let left = value(c, left_node)
    let right = value(c, right_node)
    if is_any(left) { return u.t_any }

    let present = narrow(u, left, u.t_null, false) ?? left
    if left.kind != TyKind.Null and not is_optional(left) {
        # Not an error — `a ?? b` on a non-optional is merely dead code.
        let got = render(left)
        let d = c.warn(
            W0102_UNREACHABLE_CODE,
            span_of(right_node),
            f"the right operand of `??` is unreachable: `{got}` is never null"
        )
        d.with_help("remove the `??`, or make the left operand optional")
    }
    return join(u, present, right)
}

fn _check_index(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let object = node.child("object")
    let subscript = node.child("index")

    # `Box[int](21)` applies type arguments; erasure makes the result the bare
    # declared type. The resolver settled which reading the brackets have — it is
    # the only pass that can (spec §8).
    if c.resolution.is_type_args(node) {
        let base = value(c, object)
        c.set_type(subscript, u.t_any)
        return base
    }

    if object is not null and object.kind == NodeKind.Ident {
        let symbol = c.resolution.symbol_of(object)
        if symbol is not null and _names_a_type(c, symbol) {
            var base = c.symbol_type(symbol)
            if base is null { base = c.lookup_type_name(c, symbol.name) }
            let resolved = base ?? u.t_any
            c.set_type(object, resolved)
            c.set_type(subscript, u.t_any)
            return resolved
        }
    }

    let target = value(c, object)
    let subscript_type = value(c, subscript)
    if is_any(target) or is_never(target) { return u.t_any }

    # `__getitem__` is a member, so the declaration has to be laid out first.
    let index_decl = type_decl_of(target)
    if index_decl is not null { c.layout_decl(c, index_decl) }

    # A constant subscript answers with the member it selects: `pair[0]` on
    # `(str, int)` is a `str`. A computed index keeps the join.
    if target.kind == TyKind.Tuple {
        let key = const_eval(subscript, null, c.consts)
        if key.kind == ConstKind.Int {
            let written = key.as_int()
            let at = if written < 0 { written + target.arg_count() } else { written }
            if at < 0 or at >= target.arg_count() {
                c.error(
                    E0404_NOT_INDEXABLE,
                    node.span,
                    f"index {written} is out of range for a {target.arg_count()}-member tuple"
                )
                return u.t_any
            }
            return target.args[at]
        }
    }

    let indexable = is_indexable(u, target)
    if indexable is not null {
        let (want_index, result) = indexable
        if not is_any(subscript_type) {
            let _ok = c.require_assignable(
                c,
                subscript,
                subscript_type,
                want_index,
                E0400_TYPE_MISMATCH,
                "invalid index type"
            )
        }
        return result
    }

    let decl = type_decl_of(target)
    if decl is not null and find_member(decl, "__getitem__") is not null { return u.t_any }
    if decl is null and (target.kind == TyKind.Class or target.kind == TyKind.Trait) {
        return u.t_any
    }

    let got = render(target)
    let d = c.error(E0404_NOT_INDEXABLE, node.span, f"`{got}` cannot be indexed")
    if decl is not null { d.with_help(f"define `__getitem__` on `{got}`") }
    return u.t_any
}

# The names whose brackets mean a generic instantiation rather than a subscript.
fn _names_a_type(c: Ctx, symbol: Symbol) -> bool {
    if is_type_symbol(symbol.kind) or symbol.kind == SymbolKind.TypeAlias { return true }
    return c.ident_type_decl(symbol) is not null
}

#: The type of `object[start:stop:step]`. A list, a `str`, `bytes` and a tuple
#: slice to themselves; a user type needs `__getitem__`. Every bound that is
#: written must be an `int` or `null`.
pub fn slice(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let target = value(c, node.child("object"))

    var parts: list[Node] = []
    for name in ["start", "stop", "step"] {
        let part = node.child(name)
        if part is not null { parts.push(part) }
    }
    for part in parts {
        let t = value(c, part)
        if is_any(t) or t.kind == TyKind.Int or t.kind == TyKind.Null { continue }
        let _ok = c.require_assignable(
            c,
            part,
            t,
            u.t_int,
            E0400_TYPE_MISMATCH,
            "a slice bound must be an int"
        )
    }

    if is_any(target) or is_never(target) { return u.t_any }
    match target.kind {
        TyKind.List | TyKind.Str | TyKind.Bytes | TyKind.Tuple => { return target }
        _ => { }
    }

    let decl = type_decl_of(target)
    if decl is not null and find_member(decl, "__getitem__") is not null { return u.t_any }
    if decl is null and target.kind == TyKind.Class { return u.t_any }

    let got = render(target)
    c.error(E0404_NOT_INDEXABLE, node.span, f"`{got}` cannot be sliced")
    return u.t_any
}

fn _check_range(c: Ctx, node: Node) -> Type {
    let u = c.universe
    var ends: list[Node] = []
    for name in ["start", "stop"] {
        let end = node.child(name)
        if end is not null { ends.push(end) }
    }
    for end in ends {
        let t = value(c, end)
        if is_any(t) or t.kind == TyKind.Int { continue }
        let _ok = c.require_assignable(
            c,
            end,
            t,
            u.t_int,
            E0400_TYPE_MISMATCH,
            "a range bound must be an int"
        )
    }
    return u.t_range
}

fn _check_fstring(c: Ctx, node: Node) -> Type {
    let u = c.universe
    for part in node.child_list("parts") {
        if part.kind == NodeKind.StrLit {
            c.set_type(part, u.t_str)
            continue
        }
        value(c, part)
    }
    return u.t_str
}

#: Whether a member declared with `vis` on `owner` is reachable from here.
#: `protected` reaches subclasses and `private` only the declaring class, which
#: is `Ctx.current_class` — outside a class body only the public surface shows.
pub fn visible_from(c: Ctx, owner: TypeDecl?, vis: int) -> bool {
    if vis == VIS_PUBLIC or owner is null { return true }
    let source = c.current_class
    if source is null { return false }
    if vis == VIS_PROTECTED { return is_subclass_of(source, owner) }
    return source == owner
}

#: How a visibility reads in prose.
pub fn visibility_word(vis: int) -> str {
    return if vis == VIS_PROTECTED { "protected" } else { "private" }
}

#: The declaration behind `Account` in `Account.total()`, as opposed to an
#: instance of it. Stamps the name's own type on the way past.
pub fn static_target_decl(c: Ctx, object: Node?) -> TypeDecl? {
    if object is null or object.kind != NodeKind.Ident { return null }
    let decl = c.ident_type_decl(c.resolution.symbol_of(object))
    if decl is not null { c.set_type(object, decl_type(c, decl)) }
    return decl
}

# Resolve `name` against a declared type. `static_access` distinguishes
# `Account.count` from `account.count`; enum variants live in the field list.
fn _lookup_member(
    c: Ctx,
    site: Node,
    decl: TypeDecl,
    name: str,
    static_access: bool
) -> (Type, Node?) {
    let u = c.universe

    let field = find_field(decl, name)
    if field is not null {
        let owner = owner_of_field(decl, name)
        if not visible_from(c, owner, field.visibility) {
            let owner_name = if owner is null { decl.name } else { owner.name }
            let word = visibility_word(field.visibility)
            let d = c.error(
                E0701_PRIVATE_ACCESS,
                site.span,
                f"`{name}` is {word} to `{owner_name}`"
            )
            if owner is not null { d.label(owner.span, "declared here") }
            d.with_help("mark it `pub` or add an accessor method")
        }
        return (field.ty ?? u.t_any, null)
    }

    var found: MemberInfo? = find_accessor(decl, name, false)
    if found is null { found = find_member(decl, name) }
    if found is not null {
        let owner = owner_of_member(decl, name)
        if not visible_from(c, owner, found.visibility) {
            let owner_name = if owner is null { decl.name } else { owner.name }
            let word = visibility_word(found.visibility)
            let d = c.error(
                E0701_PRIVATE_ACCESS,
                site.span,
                f"`{name}` is {word} to `{owner_name}`"
            )
            if owner is not null { d.label(owner.span, "declared here") }
        }
        # Reaching a trait's default method through a concrete receiver means
        # `Self` is that receiver. Unsubstituted, the call guarded `Self -> Cents`.
        return (_substitute_self(c, found.ty, decl) ?? u.t_any, found.node)
    }

    # A parent this unit cannot see may well declare the member: saying "no
    # member" cost every subclass of an imported class its inherited surface.
    if inherits_opaquely(decl) { return (u.t_any, null) }

    let got = render(decl.ty)
    let d = c.error(E0410_UNKNOWN_MEMBER, site.span, f"`{got}` has no member `{name}`")
    d.label(decl.span, f"`{decl.name}` is declared here")
    suggest_member(d, decl, name)
    if static_access and decl.is_enum {
        let plural = if decl.variant_count == 1 { "" } else { "s" }
        d.with_help(f"`{decl.name}` has {decl.variant_count} variant{plural}")
    }
    return (u.t_any, null)
}

#: The type of `object.name`, and the declaration node behind it when there is
#: one — the C's `AstNode **outDecl`, which is what lets a call through a method
#: be bound against a real parameter list. Null for a field or a dynamic load.
#: `?.` narrows `null` off the receiver and puts it back on the result.
# `Self` inside an inherited signature, bound to the type that inherited it.
fn _substitute_self(c: Ctx, ty: Type?, decl: TypeDecl?) -> Type? {
    if ty is null or decl is null { return ty }
    let concrete = decl.ty
    # Not inside the trait itself: there `Self` is still the open parameter, and
    # substituting turned `fn(int) -> Self` into `fn(int) -> Scalable`.
    if concrete is null or decl.is_trait { return ty }
    if not mentions_generic_param(ty) { return ty }
    # Spelled out rather than imported: `decl.jai` owns `SELF_NAME` and imports
    # this file for `Ctx`, so taking it from there would close a cycle.
    return substitute(c.universe, ty, {"Self": concrete})
}

pub fn member(c: Ctx, node: Node) -> (Type, Node?) {
    let u = c.universe
    let object = node.child("object")
    let name = node.opt_text("name") ?? ""
    let optional = node.kind == NodeKind.OptMember

    let static_decl = static_target_decl(c, object)
    if static_decl is not null {
        c.layout_decl(c, static_decl)
        return _lookup_member(c, node, static_decl, name, true)
    }

    if object is not null and object.kind == NodeKind.Super {
        let owner = c.current_class
        let parent = if owner is null { null } else { owner.superclass }
        c.set_type(object, decl_type(c, parent))
        if parent is null { return (u.t_any, null) }
        return _lookup_member(c, node, parent, name, false)
    }

    var target = value(c, object)

    # `import m` binds a module object and `m.f` is resolved by name at run time.
    # The signature loader knows what `m` declares, so the call is still checked.
    if c.is_module(c, object) {
        let (result, decl_node) = c.module_member(c, object, name)
        return (if optional { optional_of(u, result) } else { result }, decl_node)
    }

    if optional {
        let present = narrow(u, target, u.t_null, false)
        if present is not null { target = present }
    }
    if is_any(target) or is_never(target) { return (u.t_any, null) }

    # Builtin containers, modules and Error classes carry methods in the runtime.
    let decl = type_decl_of(target)
    if decl is null { return (u.t_any, null) }
    c.layout_decl(c, decl)

    if is_optional(target) and not optional {
        let got = render(target)
        let d = c.error(
            E0400_TYPE_MISMATCH,
            node.span,
            f"`{got}` may be null, so `.{name}` is not safe"
        )
        d.with_help(f"use `?.{name}`, or test with `!= null` first")
    }

    let (result, decl_node) = _lookup_member(c, node, decl, name, false)
    return (if optional { optional_of(u, result) } else { result }, decl_node)
}

# What is known about the thing being called. `CalleeInfo` in the C, filled in
# by `_check_call` and read by `_check_args`.
class CalleeInfo {
    #: The function declaration, when the callee is statically known.
    pub var decl: Node?
    #: The callee's type, when it is a function type.
    pub var ty: Type?
    #: What to call it in a diagnostic.
    pub var name: str?
    #: 1 when the declaration writes `self` first.
    pub var self_skip: int
    #: A named callee reports E0602 rather than E0600.
    pub var named: bool
    #: Where the value being called was bound, for an arity error with no `fn`.
    pub var binding_span: Span?
    #: What the binding is called in prose: "parameter", "local binding"...
    pub var binding_what: str?
    #: Set when that binding hides a method of the same name on the enclosing
    #: class, which is why the arity looks wrong.
    pub var shadowed_in: str?

    #: An empty record, which is the C's `memset`.
    pub fn init(self) {
        self.decl = null
        self.ty = null
        self.name = null
        self.self_skip = 0
        self.named = false
        self.binding_span = null
        self.binding_what = null
        self.shadowed_in = null
    }
}

fn _binding_word(kind: SymbolKind) -> str {
    match kind {
        SymbolKind.Param => { return "parameter" }
        SymbolKind.Local => { return "local binding" }
        SymbolKind.Upvalue => { return "captured binding" }
        SymbolKind.Global => { return "global binding" }
        SymbolKind.Field => { return "field" }
        SymbolKind.Method => { return "method" }
        SymbolKind.Class => { return "class" }
        SymbolKind.Trait => { return "trait" }
        SymbolKind.Enum => { return "enum" }
        SymbolKind.EnumVariant => { return "enum variant" }
        SymbolKind.Module => { return "module" }
        SymbolKind.TypeAlias => { return "type alias" }
        SymbolKind.GenericParam => { return "generic parameter" }
        SymbolKind.Builtin => { return "builtin" }
    }
}

# The function declaration this name *denotes*, or null. `Symbol.decl` is the
# node that DECLARES the symbol, which for a parameter or for `self` is the
# enclosing `fn`, so only the resolver's back-link says which binding names the
# function. `Symbol.importedDecl` is followed first: a `Fn` type carries arity
# and nothing else, so every default argument reads as required and a keyword
# argument has no name to be checked against, which is what E0603 needs.
fn _denoted_function_decl(c: Ctx, symbol: Symbol?) -> Node? {
    if symbol is null { return null }
    let imported = c.imported_decl(symbol)
    if imported is not null {
        return if imported.kind == NodeKind.FnDecl { imported } else { null }
    }
    let decl = symbol.decl
    if decl is null or decl.kind != NodeKind.FnDecl { return null }
    return if c.resolution.symbol_of(decl) == symbol { decl } else { null }
}

#: How many leading parameters of `fn_node` are the receiver: 1 or 0.
pub fn self_skip_of(fn_node: Node?) -> int {
    if fn_node is null or fn_node.kind != NodeKind.FnDecl { return 0 }
    if fn_node.flag("isStatic") { return 0 }
    let params = fn_node.records("params")
    if params.len() <= 0 { return 0 }
    return if param_name(params[0]) == "self" { 1 } else { 0 }
}

#: The `init` of `decl` or of the nearest ancestor that declares one, and the
#: declaration it was found on.
pub fn find_init_decl(decl: TypeDecl?) -> (Node?, TypeDecl?) {
    var at = decl
    var depth = 0
    while at is not null and depth < MAX_DECL_DEPTH {
        let found = at.find_member("init")
        if found is not null and found.node is not null { return (found.node, at) }
        at = at.superclass
        depth += 1
    }
    return (null, null)
}

fn _check_arg_expr(c: Ctx, node: Node?, want: Type?, what: str) -> void {
    if want is not null { c.apply_context(c, node, want) }
    let got = value(c, node)
    if want is null { return }
    let _ok = c.require_assignable(c, node, got, want, E0400_TYPE_MISMATCH, what)
}

# Bind arguments to parameters and type-check each one, exactly once each.
fn _check_args(c: Ctx, call: Node, info: CalleeInfo) -> void {
    let u = c.universe
    let args = call.records("args")
    let arg_count = args.len()
    let arity_code = if info.named { E0602_BAD_CALL_ARITY } else { E0600_ARITY_MISMATCH }
    let name = info.name ?? "this callable"

    # Per-argument expected type and bound parameter name, for the final pass.
    let tracked = if arg_count < MAX_ARGS { arg_count } else { MAX_ARGS }
    var targets: list[Type?] = []
    var bound: list[str?] = []
    var slot_index = 0
    while slot_index < tracked {
        targets.push(null)
        bound.push(null)
        slot_index += 1
    }

    var saw_keyword = false
    var saw_spread = false
    for (position, given) in args.enumerate() {
        if arg_spread(given) {
            saw_spread = true
            continue
        }
        let given_name = arg_name(given)
        if given_name is null {
            if saw_keyword {
                c.error(
                    E0605_POSITIONAL_AFTER_KEYWORD,
                    arg_span(given),
                    "positional argument follows a keyword argument"
                )
            }
            continue
        }
        saw_keyword = true
        var earlier = 0
        while earlier < position {
            if arg_name(args[earlier]) == given_name {
                let d = c.error(
                    E0604_DUPLICATE_ARGUMENT,
                    arg_span(given),
                    f"argument `{given_name}` is given twice"
                )
                d.label(arg_span(args[earlier]), "first given here")
                break
            }
            earlier += 1
        }
    }

    let decl_node = info.decl
    var params: list[dict[str, any]] = []
    var param_count = 0
    var declared = false
    if decl_node is not null and decl_node.kind == NodeKind.FnDecl {
        declared = true
        let written = decl_node.records("params")
        param_count = written.len() - info.self_skip
        if param_count < 0 { param_count = 0 }
        if param_count > 0 { params = written[info.self_skip:] }
    }

    # Resolved once, up front: a declaration in another module has to be read in
    # *its* namespace, while the loop below reports in this file's.
    var param_types: list[Type] = []
    if declared and param_count <= MAX_ARGS {
        let saved = c.enter_foreign(c.origin_of_node(decl_node))
        for parameter in params {
            let written = param_type(parameter)
            var resolved: Type? = u.t_any
            if written is not null { resolved = c.resolve_ast_type(c, written) }
            param_types.push(resolved ?? u.t_any)
        }
        c.leave_foreign(saved)
    }

    if declared and param_count <= MAX_ARGS {
        var filled: list[bool] = []
        var at = 0
        while at < param_count {
            filled.push(false)
            at += 1
        }
        var variadic = -1
        var kw_rest = -1
        for (position, parameter) in params.enumerate() {
            if param_variadic(parameter) { variadic = position }
            if param_kwrest(parameter) { kw_rest = position }
        }

        var positional = 0
        var reported_overflow = false
        for (position, given) in args.enumerate() {
            if arg_spread(given) { continue }
            let given_name = arg_name(given)

            if given_name is null {
                var slot = -1
                if variadic >= 0 and positional >= variadic {
                    slot = variadic
                } elif positional < param_count and not param_kwrest(params[positional]) {
                    slot = positional
                }

                if slot < 0 {
                    if not saw_spread and not reported_overflow {
                        let plural = if param_count == 1 { "" } else { "s" }
                        let were = if arg_count == 1 { "was" } else { "were" }
                        let d = c.error(
                            arity_code,
                            arg_span(given),
                            f"`{name}` takes {param_count} argument{plural} but {arg_count} {were} given"
                        )
                        if decl_node is not null {
                            d.label(decl_node.span, f"`{name}` is declared here")
                        }
                        reported_overflow = true
                    }
                } else {
                    if slot != variadic { filled[slot] = true }
                    if position < tracked {
                        targets[position] = param_types[slot]
                        bound[position] = param_name(params[slot])
                    }
                }
                positional += 1
                continue
            }

            var slot = -1
            for (candidate, parameter) in params.enumerate() {
                if param_variadic(parameter) or param_kwrest(parameter) { continue }
                if param_name(parameter) == given_name {
                    slot = candidate
                    break
                }
            }

            if slot < 0 {
                if kw_rest < 0 {
                    let d = c.error(
                        E0603_UNKNOWN_KEYWORD_ARG,
                        arg_span(given),
                        f"`{name}` has no parameter named `{given_name}`"
                    )
                    var best: str? = null
                    var best_distance = SUGGEST_NO_MATCH
                    for parameter in params {
                        let distance = closer(given_name, param_name(parameter), best_distance)
                        if distance is null { continue }
                        best = param_name(parameter)
                        best_distance = distance
                    }
                    if best is not null { d.with_help(f"did you mean `{best}`?") }
                }
                continue
            }

            if filled[slot] {
                c.error(
                    E0604_DUPLICATE_ARGUMENT,
                    arg_span(given),
                    f"argument `{given_name}` is given twice"
                )
            }
            filled[slot] = true
            if position < tracked {
                targets[position] = param_types[slot]
                bound[position] = param_name(params[slot])
            }
        }

        if not saw_spread {
            var missing = 0
            var first_missing: str? = null
            for (position, parameter) in params.enumerate() {
                if filled[position] { continue }
                if param_variadic(parameter) or param_kwrest(parameter) { continue }
                if param_default(parameter) is not null { continue }
                if first_missing is null { first_missing = param_name(parameter) }
                missing += 1
            }
            if missing > 0 {
                let plural = if missing == 1 { "" } else { "s" }
                let d = c.error(
                    arity_code,
                    call.span,
                    f"`{name}` is missing {missing} required argument{plural}"
                )
                if first_missing is not null { d.with_help(f"the first one is `{first_missing}`") }
                if decl_node is not null { d.label(decl_node.span, f"`{name}` is declared here") }
            }
        }
    } elif info.ty is not null and info.ty.kind == TyKind.Fn and not saw_spread and not saw_keyword {
        let signature = info.ty
        let want = signature.arg_count()
        let variadic = signature.is_variadic()
        if arg_count != want and not (variadic and arg_count >= want - 1) {
            let plural = if want == 1 { "" } else { "s" }
            let were = if arg_count == 1 { "was" } else { "were" }
            let d = c.error(
                arity_code,
                call.span,
                f"`{name}` takes {want} argument{plural} but {arg_count} {were} given"
            )
            # Show the signature that was actually checked, at its binding.
            let binding_span = info.binding_span
            let binding_what = info.binding_what ?? "binding"
            if info.binding_what is not null and binding_span is not null {
                if binding_span.is_known() {
                    let rendered = render(signature)
                    d.label(
                        binding_span,
                        f"`{name}` is the {binding_what} declared here, of type `{rendered}`"
                    )
                }
            }
            let shadowed = info.shadowed_in
            if shadowed is not null {
                let hint = f"which shadows the method `{shadowed}.{name}`"
                let fix = f"write `self.{name}(...)` to call the method"
                d.with_help(f"this `{name}` is the {binding_what}, {hint}; {fix}")
            }
        }
        var position = 0
        while position < tracked and position < want {
            targets[position] = signature.args[position]
            position += 1
        }
    }

    # One difference from the C: it truncates this label into a 96-byte buffer.
    for (position, given) in args.enumerate() {
        let held = arg_value(given)
        if position >= tracked {
            value(c, held)
            continue
        }
        var label = f"argument {position + 1} of `{name}`"
        let bound_name = bound[position]
        if bound_name is not null { label = f"argument `{bound_name}` of `{name}`" }
        let want = if arg_spread(given) { null } else { targets[position] }
        _check_arg_expr(c, held, want, label)
    }
}

fn _check_construct(c: Ctx, call: Node, decl: TypeDecl) -> Type {
    c.layout_decl(c, decl)

    if decl.is_trait {
        c.error(
            E0710_ABSTRACT_INSTANTIATION,
            call.span,
            f"`{decl.name}` is a trait and cannot be instantiated"
        )
    } elif decl.is_abstract {
        let d = c.error(
            E0710_ABSTRACT_INSTANTIATION,
            call.span,
            f"`{decl.name}` is abstract and cannot be instantiated"
        )
        d.label(decl.span, "declared abstract here")
        for member_info in decl.members {
            if member_info.is_abstract {
                d.with_help(f"`{member_info.name}` has no implementation")
                break
            }
        }
    }

    let info = CalleeInfo()
    info.name = decl.name
    info.named = true
    let (init_decl, _owner) = find_init_decl(decl)
    info.decl = init_decl
    info.self_skip = self_skip_of(init_decl)
    if init_decl is null and call.records("args").len() > 0 {
        c.error(
            E0602_BAD_CALL_ARITY,
            call.span,
            f"`{decl.name}` has no `init`, so it takes no arguments"
        )
    }
    _check_args(c, call, info)
    return decl_type(c, decl)
}

fn _check_call(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let callee = node.child("callee")
    if callee is null { return u.t_any }

    # ClassName(...) constructs; there is no `new`.
    if callee.kind == NodeKind.Ident {
        let decl = c.ident_type_decl(c.resolution.symbol_of(callee))
        if decl is not null and not decl.is_enum {
            c.set_type(callee, decl_type(c, decl))
            return _check_construct(c, node, decl)
        }
    }

    if callee.kind == NodeKind.Super {
        let owner = c.current_class
        let parent = if owner is null { null } else { owner.superclass }
        c.set_type(callee, decl_type(c, parent))
        let info = CalleeInfo()
        info.name = "super"
        info.named = true
        if parent is not null {
            let (init_decl, _owner) = find_init_decl(parent)
            info.decl = init_decl
            info.self_skip = self_skip_of(init_decl)
            info.name = parent.name
        }
        _check_args(c, node, info)
        return u.t_void
    }

    var callee_decl: Node? = null
    var fn_type: Type? = null
    # `a?.m(x)` short-circuits the whole call, not just the member load (spec
    # §4.2), so the optional is taken off the callee and put back on the result.
    let opt_chain = callee.kind == NodeKind.OptMember

    if callee.kind == NodeKind.Member or opt_chain {
        let (loaded, loaded_decl) = member(c, callee)
        fn_type = loaded
        callee_decl = loaded_decl
        if opt_chain {
            let present = narrow(u, fn_type, u.t_null, false)
            if present is not null { fn_type = present }
        }
        c.set_type(callee, fn_type)
        # `m.Thing(...)` after `import m` names a class: the member load yielded
        # the declaration itself, so this is construction and not a call.
        if c.is_module(c, callee.child("object")) {
            let class_decl = type_decl_of(fn_type)
            if class_decl is not null and not class_decl.is_enum {
                return _check_construct(c, node, class_decl)
            }
        }
    } else {
        fn_type = value(c, callee)
        if callee.kind == NodeKind.Ident {
            callee_decl = _denoted_function_decl(c, c.resolution.symbol_of(callee))
        }
    }

    let info = CalleeInfo()
    info.decl = callee_decl
    if fn_type is not null and fn_type.kind == TyKind.Fn { info.ty = fn_type }
    info.self_skip = self_skip_of(callee_decl)

    if callee.kind == NodeKind.Ident {
        info.name = callee.opt_text("name")
        info.named = true
        # A name that denotes no declaration is a value being called; its binding
        # is recorded so an arity error can show the signature it checked.
        let symbol = c.resolution.symbol_of(callee)
        if callee_decl is null and symbol is not null {
            info.binding_span = symbol.decl_span
            info.binding_what = _binding_word(symbol.kind)
            # Members are copied down during layout, so one lookup covers all.
            let enclosing = c.current_class
            if enclosing is not null and _is_frame_binding(symbol.kind) {
                if find_member(enclosing, symbol.name) is not null {
                    let owner = owner_of_member(enclosing, symbol.name)
                    if owner is not null { info.shadowed_in = owner.name }
                }
            }
        }
    } elif callee.kind == NodeKind.Member or opt_chain {
        info.name = callee.opt_text("name")
        info.named = callee_decl is not null
    }

    if fn_type is not null and fn_type.kind == TyKind.Fn {
        _check_args(c, node, info)
        let result = fn_type.ret ?? u.t_any
        return if opt_chain { optional_of(u, result) } else { result }
    }

    if is_any(fn_type) or is_never(fn_type) {
        info.decl = null
        info.ty = null
        _check_args(c, node, info)
        return u.t_any
    }

    let decl = type_decl_of(fn_type)
    if decl is not null { c.layout_decl(c, decl) }
    if decl is not null and find_member(decl, "__call__") is not null {
        info.decl = null
        info.ty = null
        _check_args(c, node, info)
        return u.t_any
    }
    if is_callable(fn_type) {
        info.decl = null
        info.ty = null
        _check_args(c, node, info)
        return u.t_any
    }

    let got = render(fn_type)
    let d = c.error(E0403_NOT_CALLABLE, node.span, f"`{got}` is not callable")
    if decl is not null { d.with_help(f"define `__call__` on `{got}`") }
    info.decl = null
    info.ty = null
    _check_args(c, node, info)
    return u.t_any
}

# The three symbol kinds that occupy a frame slot and so can shadow a method.
fn _is_frame_binding(kind: SymbolKind) -> bool {
    if kind == SymbolKind.Param or kind == SymbolKind.Local { return true }
    return kind == SymbolKind.Upvalue
}

fn _check_comprehension(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let comp_kind: CompKind = node.get("compKind")
    var shape = TyKind.List
    if comp_kind == CompKind.Set {
        shape = TyKind.Set
    } elif comp_kind == CompKind.Dict {
        shape = TyKind.Dict
    }
    let arity = if shape == TyKind.Dict { 2 } else { 1 }
    let want = _take_container_context(c, node, shape, arity)

    let mark = c.narrow_mark()
    for clause in node.records("clauses") {
        let over = clause_iterable(clause)
        let source = value(c, over)
        let elem = iterable_element(c, over, source)
        c.check_pattern(c, clause_pattern(clause), elem)
        for filter in clause_conditions(clause) {
            condition(c, filter, "a comprehension filter")
        }
    }

    let element = node.child("element")
    let value_type = if element is null { u.t_any } else { value(c, element) }
    let key_node = node.child("keyExpr")
    var key_type: Type? = null
    if key_node is not null { key_type = value(c, key_node) }
    c.narrow_restore(mark)

    # Annotated: report the element that does not fit, not the whole literal.
    if want is not null {
        if key_node is not null {
            let _key_ok = c.require_assignable(
                c,
                key_node,
                key_type ?? u.t_any,
                want.args[0],
                E0400_TYPE_MISMATCH,
                "mismatched key type"
            )
        }
        let _ok = c.require_assignable(
            c,
            element,
            value_type,
            want.args[want.arg_count() - 1],
            E0400_TYPE_MISMATCH,
            "mismatched element type"
        )
        return want
    }

    match comp_kind {
        CompKind.List => { return list_of(u, value_type) }
        CompKind.Set => { return set_of(u, value_type) }
        CompKind.Dict => { return dict_of(u, key_type ?? u.t_any, value_type) }
        CompKind.Generator => { return u.t_any }
    }
}

fn _check_function_expr(c: Ctx, node: Node) -> Type {
    c.check_function(c, node)
    let t = c.function_type(c, node)
    c.set_symbol_type(c.resolution.symbol_of(node), t)
    return t
}

fn _check_if_expr(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let cond = node.child("cond")
    condition(c, cond, "an `if`")

    let facts = c.collect_facts(cond, false)
    var mark = c.narrow_mark()
    c.narrow_apply(facts, false)
    let then_type = expr(c, node.child("thenBranch"))
    c.narrow_restore(mark)

    let else_branch = node.child("elseBranch")
    if else_branch is null { return optional_of(u, then_type) }

    let else_facts = c.collect_facts(cond, true)
    mark = c.narrow_mark()
    c.narrow_apply(else_facts, false)
    let else_type = expr(c, else_branch)
    c.narrow_restore(mark)
    return join(u, then_type, else_type)
}

fn _check_cast(c: Ctx, node: Node) -> Type {
    let u = c.universe
    # A cast this checker inserted carries its target in the type table: a
    # Jaithon `TypeExpr` has no `resolved` field for `wrapInCast` to fill in.
    let want = c.resolve_ast_type(c, node.type_field("target")) ?? c.type_of(node)
    value(c, node.child("operand"))
    return want ?? u.t_any
}

#: The type of `node`, recorded for it and folded where it folds. The answer is
#: recorded through `Ctx.set_type` before folding, because `fold.try_fold` reads
#: the checked type to decide whether the literal it produced is more precise.
pub fn expr(c: Ctx, node: Node?) -> Type {
    if node is null { return c.universe.t_any }
    let u = c.universe
    var t: Type? = null

    match node.kind {
        NodeKind.IntLit => { t = u.t_int }
        NodeKind.FloatLit => { t = u.t_float }
        NodeKind.StrLit => { t = u.t_str }
        NodeKind.BoolLit => { t = u.t_bool }
        NodeKind.NullLit => { t = u.t_null }
        NodeKind.FString => { t = _check_fstring(c, node) }

        NodeKind.Ident => {
            t = _check_ident(c, node)
            # `_check_ident` may have folded the node into a literal.
            if node.kind != NodeKind.Ident { return c.type_of(node) ?? u.t_any }
        }

        NodeKind.Self => {
            let owner = c.current_class
            t = if owner is null { u.t_any } else { decl_type(c, owner) }
        }

        NodeKind.Super => {
            let owner = c.current_class
            let parent = if owner is null { null } else { owner.superclass }
            t = if parent is null { u.t_any } else { decl_type(c, parent) }
        }

        NodeKind.ListLit | NodeKind.SetLit | NodeKind.TupleLit => {
            t = _check_sequence_literal(c, node)
        }

        NodeKind.DictLit => { t = _check_dict_literal(c, node) }
        NodeKind.Unary => { t = _check_unary(c, node) }
        NodeKind.Binary => { t = _check_binary(c, node) }
        NodeKind.Logical => { t = _check_logical(c, node) }
        NodeKind.CompareChain => { t = _check_chain(c, node) }
        NodeKind.Ternary => { t = _check_ternary(c, node) }
        NodeKind.Coalesce => { t = _check_coalesce(c, node) }
        NodeKind.Call => { t = _check_call(c, node) }
        NodeKind.Index => { t = _check_index(c, node) }
        NodeKind.Slice => { t = slice(c, node) }

        NodeKind.Member | NodeKind.OptMember => {
            let (loaded, _decl) = member(c, node)
            t = loaded
        }

        NodeKind.Lambda | NodeKind.AnonFn => { t = _check_function_expr(c, node) }
        NodeKind.Comprehension => { t = _check_comprehension(c, node) }
        NodeKind.Range => { t = _check_range(c, node) }
        NodeKind.IfExpr => { t = _check_if_expr(c, node) }
        NodeKind.MatchExpr => { t = c.match_expr(c, node, false) }
        NodeKind.Cast => { t = _check_cast(c, node) }

        NodeKind.Yield | NodeKind.Await => {
            let operand = node.child("operand")
            if operand is not null { value(c, operand) }
            t = u.t_any
        }

        # A block in expression position is a lambda body or an `if` arm.
        NodeKind.Block => {
            c.check_block(c, node)
            t = u.t_any
        }

        _ => {
            if is_statement(node.kind) {
                c.check_stmt(c, node)
                t = u.t_void
            } else {
                t = u.t_any
            }
        }
    }

    c.set_type(node, t ?? u.t_any)
    if c.fold_constants {
        let (_folded, after) = try_fold(u, node, c.type_of(node), null, c.consts)
        c.set_type(node, after)
    }
    return c.type_of(node) ?? u.t_any
}
