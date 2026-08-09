#: Declarations: class, trait and enum layout, conformance, and bodies.
#:
#: A port of `src/sema/check_decl.c`, with the three declaration forms
#: `check_stmt.c` owns that nothing else in this directory can answer for —
#: `jaiChkFunctionType`, `jaiChkFunction` and `checkVarDecl`. They are here
#: because a method's signature *is* a member's type and a field default *is* an
#: initialiser check: splitting them off would leave layout calling into a file
#: that does not exist and reproducing half of it in the meantime.
#:
#: The work is three stages. **Layout** turns a declaration node into a
#: `TypeDecl`: it resolves the superclass and the traits, assigns field slots,
#: and installs each declared method, getter and setter. **Conformance** checks
#: that a class implements every requirement its traits name, substituting the
#: trait's own generic arguments and `Self` before comparing signatures.
#: **Bodies** checks each method, field default and variant payload against the
#: type the first two stages built.
#:
#: ## Layout builds; `nominal.jai` reads. Nothing is copied.
#:
#: This is the one decision in the file that had to be made rather than ported,
#: and it goes the other way from the C.
#:
#: The C's layout **copies**: a class's `fields` array begins with its parent's,
#: its `members` array begins with its parent's and then its traits', and every
#: later lookup is a linear scan of one flat array. `nominal.jai` was written
#: for the opposite arrangement — `find_field` walks the superclass chain and
#: `find_member` walks own, then traits, then superclass, and its module doc
#: states outright that nothing has copied anything. Both cannot be true. This
#: port keeps `nominal.jai`'s: **a class's `fields` and `members` hold exactly
#: what the class itself declares**, and every inherited answer is walked for.
#:
#: Three consequences, all deliberate:
#:
#: - **Slots still run parents first.** `_next_slot` walks the superclass chain
#:   for the highest non-static slot and numbers this class's own fields after
#:   it, so an inherited field keeps its slot and a subclass's inline caches
#:   stay valid (§6). That is the only thing the copy was load-bearing for.
#: - **The checks consult the walks.** A duplicate is a name declared twice in
#:   *this* declaration's own text; an override is measured against
#:   `find_member` on the superclass; conformance asks a three-step lookup —
#:   own text, then the superclass chain, then the traits in written order —
#:   which is the precedence the C's flat array encodes by construction.
#: - **`suggest_member` sees less.** `expr.jai`'s "did you mean" scans
#:   `decl.fields` and `decl.members` directly, so it no longer offers an
#:   inherited name. That is a weaker help line, not a wrong one.
#:
#: **A trait is the exception, and copies.** `layout_trait` restates a
#: supertrait's members into the extending trait, exactly as the C does, because
#: a requirement set is *enumerated* rather than looked up — `verify_traits`
#: iterates `trait.members` and `nominal.jai` has no walking enumerator for it.
#: `trait Sorted[T]: Comparable[T]` therefore carries `compare(T)` in its own
#: parameters, which is also what makes the restatement correct rather than
#: merely present.
#:
#: ## What is not implemented, and what it answers today
#:
#: | form | today |
#: |---|---|
#: | `jaiChkForeignBegin` / `jaiChkForeignEnd` | not ported: a `TypeDecl` has no module of origin, so a declaration read from another module is laid out as if local and its diagnostics are **reported rather than suppressed** |
#: | `DeclEntry.alias` and `DeclEntry.visible` | not ported: `Ctx.register_decl` keys on the declared name alone, so an aliased or hidden import is not modelled |
#: | a method body, a field default's statements | `Ctx.check_block`, whose default does nothing until `stmt.jai` lands |
#: | a `let` pattern's bindings | `Ctx.check_pattern`, whose default does nothing |
#: | every written annotation | `Ctx.resolve_ast_type`, whose default answers null — so a superclass and a trait both read as *opaque by name*, which is the C's answer for a type it cannot resolve |
#: | E0407 generic arity on a trait reference | `resolve_ast_type`'s, not this file's; `_trait_binding_of` declines to bind a mismatched arity rather than reporting a second time |
#: | `Symbol.isConstFolded` / `Symbol.constValue` | a Jaithon `Symbol` has neither, so a folded `const` is recorded in `Ctx.consts` under its name, which is where `fold.jai` and `expr.jai` read it |
#: | `TYPE_BUF` truncation | not reproduced: the C renders into fixed buffers and truncates, and nothing here does |
#:
#: Two E0709 corners come out **stricter** than the C, both because the C reads
#: one flat array where this reads three lists:
#:
#: - Redeclaring a method a class also inherits is reported here. The C's
#:   `existing` lands in the parent region, which its duplicate test excuses,
#:   so the second declaration silently replaces the first.
#: - A field and a method of one name, where a trait also declares that name,
#:   is reported here. The C finds the trait's entry, skips the field test, and
#:   accepts the class.
#:
#: ```
#: let c = Ctx(Universe())
#: install(c)
#: let node = Node(NodeKind.ClassDecl, {"name": "Point"})
#: register_decl(c, node)?.name        # "Point"
#: c.decl_named("Point")?.is_trait     # false
#: ```
from jaithon.ast import DeclKind, Node, NodeKind, Span, TypeExpr, TypeKind, Visibility
from jaithon.ast import always_diverges, preorder
from jaithon.compile.check.assign import is_subtype
from jaithon.compile.check.expr import Ctx, E0400_TYPE_MISMATCH, VIS_PRIVATE, VIS_PROTECTED
from jaithon.compile.check.expr import VIS_PUBLIC, find_init_decl, self_skip_of, value
from jaithon.compile.check.fold import const_eval, const_replace
from jaithon.compile.check.kinds import FN_KWREST, FN_VARIADIC, MAX_DECL_DEPTH, TyKind
from jaithon.compile.check.nominal import find_accessor, find_field, find_member
from jaithon.compile.check.predicate import is_optional
from jaithon.compile.check.render import render
from jaithon.compile.check.substitute import substitute
from jaithon.compile.check.ty import FieldInfo, MemberInfo, Type, TypeDecl
from jaithon.compile.check.universe import Universe, dict_of, fn_of, generic_param
from jaithon.compile.check.universe import list_of, named
from jaithon.compile.resolve import Symbol

# ------------------------------------------------------------------
# Diagnostic codes
# ------------------------------------------------------------------

#: The initialiser of a `const` is not a constant expression.
#:
#: Spelled here rather than in `jaithon.compile.resolve.SemaCode` for the reason
#: `expr.jai` gives for its own groups: that table is the resolver's and stops
#: where the resolver does. Both the code and the message text are a contract
#: with the C front end — Phase 4 turns `tests/errors` into goldens.
pub const E0303_NON_CONSTANT_CONST: str = "E0303"

#: A `const` was declared with no initialiser.
pub const E0304_MISSING_INITIALIZER: str = "E0304"

#: An expression-bodied function produces something its annotation refuses.
pub const E0408_RETURN_TYPE_MISMATCH: str = "E0408"

#: An annotated function can fall off its end without returning.
pub const E0601_MISSING_RETURN: str = "E0601"

#: A name in a `class` or `trait` header does not denote one.
pub const E0700_UNKNOWN_CLASS: str = "E0700"

#: An instance method that does not take `self`.
pub const E0703_MISSING_SELF: str = "E0703"

#: An override whose signature the overridden one does not admit.
pub const E0704_INCOMPATIBLE_OVERRIDE: str = "E0704"

#: A class that names a trait it does not satisfy.
pub const E0705_TRAIT_NOT_IMPLEMENTED: str = "E0705"

#: A declaration that would, directly or through a chain, inherit from itself.
pub const E0706_CYCLIC_INHERITANCE: str = "E0706"

#: A `static fn` that takes `self`.
pub const E0707_STATIC_WITH_SELF: str = "E0707"

#: A `super(...)` that is not the first statement of `init`.
pub const E0708_SUPER_INIT_NOT_FIRST: str = "E0708"

#: One name declared twice in one class, trait or enum.
pub const E0709_DUPLICATE_MEMBER: str = "E0709"

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

#: The name `Self` is bound to inside a declaration, matching `jaiChkSelfName`.
#:
#: The parser forbids a declaration from introducing it (E0114), so binding it
#: needs no escaping and substituting it is one ordinary generic parameter.
pub const SELF_NAME: str = "Self"

# Where `_own_index` found nothing.
const _NOT_FOUND: int = -1

# `jaithon.ast.Visibility` as the `int` that `FieldInfo` and `MemberInfo` hold.
# `expr.jai` documents `VIS_*` as those ordinals; this is the table that makes
# the claim true rather than assumed.
let _VISIBILITY_ORDINAL: dict[Visibility, int] = {
    Visibility.Private: VIS_PRIVATE,
    Visibility.Protected: VIS_PROTECTED,
    Visibility.Public: VIS_PUBLIC,
}

# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

# `jaiChkIsAny`, `jaiChkIsNever`, `jaiChkIsVoid`, `jaiChkOrAny` and
# `jaiChkDeclType`. Spelled again rather than imported because `expr.jai` keeps
# its copies private, and this file must not widen that surface to reach them.
fn _is_any(t: Type?) -> bool {
    if t is null { return true }
    return t.kind == TyKind.Any or t.kind == TyKind.GenericParam
}

fn _is_never(t: Type?) -> bool {
    return t is not null and t.kind == TyKind.Never
}

fn _is_void(c: Ctx, t: Type?) -> bool {
    if t is null { return false }
    return t == c.universe.t_void and c.universe.t_void != c.universe.t_null
}

fn _or_any(c: Ctx, t: Type?) -> Type {
    return t ?? c.universe.t_any
}

fn _decl_type(c: Ctx, decl: TypeDecl?) -> Type {
    if decl is null { return c.universe.t_any }
    return decl.ty ?? c.universe.t_any
}

# `jaiChkTypeDeclOf`. The C searches its registry because a type may outlive the
# AST that declared it; interning attaches the declaration here, so the field is
# the answer.
fn _type_decl_of(t: Type?) -> TypeDecl? {
    if t is null { return null }
    if not t.is_named() { return null }
    return t.decl
}

# Typed reads of the AST sub-structure records, matching `expr.jai`'s.
fn _record_span(record: dict[str, any]) -> Span {
    let span: Span = record.get("span", Span.none())
    return span
}

fn _record_name(record: dict[str, any]) -> str? {
    let name: str? = record.get("name", null)
    return name
}

fn _record_type(record: dict[str, any]) -> TypeExpr? {
    let written: TypeExpr? = record.get("type", null)
    return written
}

fn _record_default(record: dict[str, any]) -> Node? {
    let default_value: Node? = record.get("defaultValue", null)
    return default_value
}

fn _record_flag(record: dict[str, any], name: str) -> bool {
    let flag: bool = record.get(name, false)
    return flag
}

fn _record_visibility(record: dict[str, any]) -> int {
    let written: Visibility = record.get("visibility", Visibility.Private)
    return _VISIBILITY_ORDINAL[written]
}

fn _record_params(record: dict[str, any]) -> list[dict[str, any]] {
    let params: list[dict[str, any]] = record.get("params", [])
    return params
}

fn _node_visibility(node: Node) -> int {
    return _VISIBILITY_ORDINAL[node.visibility()]
}

# ------------------------------------------------------------------
# Generic parameters and `Self`
# ------------------------------------------------------------------

#: Bind each of `generics` as a generic parameter until the enclosing
#: declaration ends.
#:
#: `jaiChkPushGenerics`. The bounds are resolved in a second pass, once every
#: parameter is in scope, so that `[T: Comparable[T]]` can see itself. A bound
#: is resolved for its diagnostics alone — nothing here records it, exactly as
#: in the C.
#:
#: ```
#: from jaithon.ast import generic
#: let c = Ctx(Universe())
#: let mark = c.type_name_mark()
#: push_generics(c, [generic("T")])
#: render(c.scoped_type_name("T"))     # "T"
#: c.type_name_restore(mark)
#: c.scoped_type_name("T") is null     # true
#: ```
pub fn push_generics(c: Ctx, generics: list[dict[str, any]]) -> void {
    for record in generics {
        let name = _record_name(record)
        c.push_type_name(name, generic_param(c.universe, name))
    }
    for record in generics {
        let bound: TypeExpr? = record.get("bound", null)
        if bound is null { continue }
        let _resolved = c.resolve_ast_type(c, bound)
    }
}

#: Bind `Self` to `ty`, or leave it abstract when `ty` is null.
#:
#: `jaiChkPushSelf`. A trait body wants the abstract form: the implementing type
#: is still unknown one level up, so `Self` stays a generic parameter and a
#: requirement written `-> Self` is substituted only when a class names the
#: trait.
#:
#: ```
#: let c = Ctx(Universe())
#: push_self(c, null)
#: render(c.scoped_type_name("Self"))     # "Self"
#: push_self(c, c.universe.t_int)
#: render(c.scoped_type_name("Self"))     # "int"
#: ```
pub fn push_self(c: Ctx, ty: Type?) -> void {
    c.push_type_name(SELF_NAME, ty ?? generic_param(c.universe, SELF_NAME))
}

# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

#: Give `node` a `TypeDecl` and an interned type, and register both.
#:
#: `jaiChkRegisterDecl`. A name that is already registered answers with the
#: declaration that holds it and registers nothing — the duplicate is `E0302`
#: and the resolver's to report. Answers null for any node that is not a class,
#: trait or enum, and for one with no name.
#:
#: ```
#: let c = Ctx(Universe())
#: let node = Node(NodeKind.EnumDecl, {"name": "Colour"})
#: register_decl(c, node)?.is_enum            # true
#: register_decl(c, node)?.name               # "Colour", the second call registers nothing
#: register_decl(c, Node.block([])) is null   # true
#: ```
pub fn register_decl(c: Ctx, node: Node?) -> TypeDecl? {
    if node is null { return null }

    var kind = TyKind.Class
    match node.kind {
        NodeKind.ClassDecl => { kind = TyKind.Class }
        NodeKind.TraitDecl => { kind = TyKind.Trait }
        NodeKind.EnumDecl => { kind = TyKind.Enum }
        _ => { return null }
    }

    let name = node.opt_text("name")
    if name is null { return null }

    let existing = c.decl_named(name)
    if existing is not null { return existing }

    let d = TypeDecl(name, node.span, node)
    d.is_trait = node.kind == NodeKind.TraitDecl
    d.is_enum = node.kind == NodeKind.EnumDecl
    d.is_abstract = node.kind == NodeKind.ClassDecl and node.flag("isAbstract")
    d.ty = named(c.universe, kind, name, d)
    c.register_decl(d)

    c.set_symbol_type(c.resolution.symbol_of(node), d.ty)
    c.set_type(node, d.ty)
    return d
}

# ------------------------------------------------------------------
# Signature compatibility
# ------------------------------------------------------------------

# An override may widen what it accepts and narrow what it returns, so one
# direction of the test is all `typeFits` ever asks.
fn _type_fits(c: Ctx, sub: Type?, super_ty: Type?) -> bool {
    if sub is null or super_ty is null { return true }
    if _is_any(sub) or _is_any(super_ty) { return true }
    if sub == super_ty { return true }
    return is_subtype(c.universe, sub, super_ty)
}

# `signatureCompatible`. Answers `(ok, why)`, where `why` is the clause the
# diagnostic reads out and is null when `ok`.
fn _signature_compatible(c: Ctx, sub: Type?, super_ty: Type?) -> (bool, str?) {
    if sub is null or super_ty is null { return (true, null) }
    if _is_any(sub) or _is_any(super_ty) { return (true, null) }

    if sub.kind != TyKind.Fn or super_ty.kind != TyKind.Fn {
        if _type_fits(c, sub, super_ty) { return (true, null) }
        return (false, "the types differ")
    }
    if sub.arg_count() != super_ty.arg_count() {
        return (false, "the parameter counts differ")
    }
    # Parameters are contravariant: the override has to accept everything the
    # overridden one did.
    for (position, accepted) in sub.args.enumerate() {
        if not _type_fits(c, super_ty.args[position], accepted) {
            return (false, "a parameter type is not general enough")
        }
    }
    if not _type_fits(c, sub.ret, super_ty.ret) {
        return (false, "the return type is not compatible")
    }
    return (true, null)
}

# ------------------------------------------------------------------
# Trait generic arguments
# ------------------------------------------------------------------

# `class MapIter[T, U]: Iterator[U]` on `trait Iterator[T]` binds the trait's
# own `T` to `U`, so its `fn next(self) -> T?` requires `-> U?` here.
#
# The C's `TraitBinding` carries the parameter names and a `Self` beside the
# values; `TypeDecl.trait_bindings` carries the values alone, because the names
# are recoverable from the trait's own declaration and `Self` is decided by
# whether the implementer is a trait.
fn _trait_binding_of(c: Ctx, trait_decl: TypeDecl, ref: TypeExpr) -> list[Type] {
    var values: list[Type] = []
    let node = trait_decl.node
    if node is null or node.kind != NodeKind.TraitDecl { return values }
    if ref.kind != TypeKind.Generic { return values }

    let generics = node.records("generics")
    # A wrong argument count is already E0407 from `resolve_ast_type`; binding a
    # prefix of it would only add a second, more confusing diagnostic.
    if generics.len() <= 0 or generics.len() != ref.args.len() { return values }

    for (position, written) in ref.args.enumerate() {
        # An argument that resolved to nothing binds the parameter to itself,
        # which is the C's "a null value leaves the parameter unchanged" —
        # `substitute` takes a `dict[str, Type]` and cannot express the null.
        let resolved = c.resolve_ast_type(c, written)
        values.push(resolved ?? generic_param(c.universe, _record_name(generics[position])))
    }
    return values
}

# The parameter names `binding` supplies values for, paired up.
fn _binding_map(trait_decl: TypeDecl, binding: list[Type]) -> dict[str, Type] {
    var out: dict[str, Type] = {}
    let node = trait_decl.node
    if node is null or binding.len() == 0 { return out }
    for (position, record) in node.records("generics").enumerate() {
        let name = _record_name(record)
        if name is null or position >= binding.len() { continue }
        out[name] = binding[position]
    }
    return out
}

# `applyTraitBinding`: the requirement as `owner` wrote it.
#
# `Self` goes first, so that a requirement written `-> Self` becomes the
# implementer's own type (§7.2). A trait extending a trait leaves it abstract,
# because the extending trait is not the implementing type either.
fn _apply_trait_binding(
    c: Ctx,
    owner: TypeDecl,
    trait_decl: TypeDecl,
    binding: list[Type],
    t: Type?
) -> Type? {
    var out = t
    if not owner.is_trait {
        let self_ty = owner.ty
        if self_ty is not null {
            out = substitute(c.universe, out, {SELF_NAME: self_ty})
        }
    }
    let bindings = _binding_map(trait_decl, binding)
    if bindings.len() == 0 { return out }
    return substitute(c.universe, out, bindings)
}

# "Iterator[(int, T)]" — the trait as this implementer wrote it, so a note can
# point back at the declaration the requirement came from.
fn _render_trait_ref(trait_decl: TypeDecl, binding: list[Type]) -> str {
    if binding.len() == 0 { return trait_decl.name }
    let args = ", ".join([render(argument) for argument in binding])
    return f"{trait_decl.name}[{args}]"
}

# The declaration the author has to write, not merely its type. Spec §7 makes
# `self` an explicit first parameter, so a help quoting the bare `fn` type would
# send the reader straight into E0703. Parameter names come from the trait, but
# the types come from `required`, which has the implementer's trait arguments
# already substituted in.
fn _render_required_method(name: str, trait_fn: Node?, required: Type?) -> str {
    var params: list[dict[str, any]] = []
    var skip = 0
    var is_static = false
    if trait_fn is not null and trait_fn.kind == NodeKind.FnDecl {
        params = trait_fn.records("params")
        skip = self_skip_of(trait_fn)
        is_static = trait_fn.flag("isStatic")
    }

    var out: list[str] = []
    if is_static { out.push("static ") }
    out.push(f"fn {name}(")

    var wrote = false
    if skip > 0 {
        out.push("self")
        wrote = true
    }

    var arg_count = 0
    if required is not null and required.kind == TyKind.Fn { arg_count = required.arg_count() }
    for position in 0..arg_count {
        var shown: str? = null
        if skip + position < params.len() { shown = _record_name(params[skip + position]) }
        let ty = render(required.args[position])
        if wrote { out.push(", ") }
        out.push(f"{shown ?? "arg"}: {ty}")
        wrote = true
    }

    var returns: Type? = null
    if required is not null { returns = required.ret }
    out.push(f") -> {render(returns)}")
    return "".join(out)
}

# ------------------------------------------------------------------
# Member lookup during layout
# ------------------------------------------------------------------

# The index in `d`'s own member list, accessor-exact first and then — when
# `fallback` — whichever accessor carries the name. `jaiChkFindAccessorIndex`
# followed by `jaiChkFindMemberIndex`, over the one list this port keeps.
fn _own_index(d: TypeDecl, name: str, want_setter: bool, fallback: bool) -> int {
    for (index, member) in d.members.enumerate() {
        if member.name == name and member.is_setter == want_setter { return index }
    }
    if not fallback { return _NOT_FOUND }
    for (index, member) in d.members.enumerate() {
        if member.name == name { return index }
    }
    return _NOT_FOUND
}

# The same question of an ancestor, through `nominal.jai`'s walks — which is
# where the members the C would have copied down actually live.
fn _inherited_pick(decl: TypeDecl?, name: str, want_setter: bool, fallback: bool) -> MemberInfo? {
    if decl is null { return null }
    let accessor = find_accessor(decl, name, want_setter)
    if accessor is not null { return accessor }
    if not fallback { return null }
    return find_member(decl, name)
}

# What the C's flat member array would hold for `name`, in the precedence its
# construction encodes: this declaration's own text, then the superclass chain,
# then the traits in the order they were written. A member reached through a
# trait comes back with that trait's arguments substituted in, because that is
# the form the C copied into the class.
fn _provided_member(c: Ctx, d: TypeDecl, name: str, want_setter: bool) -> MemberInfo? {
    let own = _own_index(d, name, want_setter, true)
    if own != _NOT_FOUND { return d.members[own] }

    let inherited = _inherited_pick(d.superclass, name, want_setter, true)
    if inherited is not null { return inherited }

    for (index, trait_decl) in d.traits.enumerate() {
        let found = _inherited_pick(trait_decl, name, want_setter, true)
        if found is null { continue }
        var binding: list[Type] = []
        if index < d.trait_bindings.len() { binding = d.trait_bindings[index] }
        let bound = _apply_trait_binding(c, d, trait_decl, binding, found.ty)
        if bound == found.ty { return found }
        return MemberInfo(
            found.name,
            ty: bound,
            visibility: found.visibility,
            is_static: found.is_static,
            is_getter: found.is_getter,
            is_setter: found.is_setter,
            is_abstract: found.is_abstract,
            node: found.node
        )
    }
    return null
}

# ------------------------------------------------------------------
# Fields
# ------------------------------------------------------------------

# The first instance slot this declaration may use: one past the highest its
# ancestors hold. The C reads it off the copied prefix; nothing is copied here,
# so the chain is walked for it.
fn _next_slot(parent: TypeDecl?) -> int {
    var next = 0
    var at = parent
    var depth = 0
    while at is not null and depth < MAX_DECL_DEPTH {
        for field in at.fields {
            if not field.is_static and field.slot >= next { next = field.slot + 1 }
        }
        at = at.superclass
        depth += 1
    }
    return next
}

# `addField`, answering the next free slot rather than advancing an out-parameter.
fn _add_field(c: Ctx, d: TypeDecl, record: dict[str, any], next_slot: int) -> int {
    let name = _record_name(record)
    if name is null { return next_slot }

    # `find_field` walks the superclass chain, so redeclaring a parent's field
    # is caught here exactly as the C catches it in the copied prefix.
    if find_field(d, name) is not null {
        let diag = c.error(
            E0709_DUPLICATE_MEMBER,
            _record_span(record),
            f"`{name}` is already declared in `{d.name}`"
        )
        diag.label(d.span, "in this declaration")
        return next_slot
    }

    let is_static = _record_flag(record, "isStatic")
    var slot = -1
    var out = next_slot
    if not is_static {
        slot = out
        out += 1
    }

    d.fields.push(
        FieldInfo(
            name,
            ty: _or_any(c, c.resolve_ast_type(c, _record_type(record))),
            slot: slot,
            visibility: _record_visibility(record),
            is_static: is_static,
            is_let: _record_flag(record, "isLet"),
            has_default: _record_default(record) is not null
        )
    )
    return out
}

# ------------------------------------------------------------------
# Members
# ------------------------------------------------------------------

# `memberTypeOf`. A property's type is what it reads or writes; a method's is
# its whole signature.
fn _member_type_of(c: Ctx, fn_node: Node, is_getter: bool, is_setter: bool) -> Type? {
    if is_getter { return declared_return_type(c, fn_node, true) }
    if is_setter {
        let params = fn_node.records("params")
        if params.len() > 0 {
            let written = _record_type(params[0])
            if written is not null { return c.resolve_ast_type(c, written) }
        }
        return c.universe.t_any
    }
    return signature_of(c, fn_node, true)
}

# `installMember`. `own_from` is the index at which this declaration's own text
# starts: zero for a class and an enum, and the end of the restated supertrait
# requirements for a trait. An entry before it is replaced, as the C replaces an
# inherited entry in place; one at or after it is a genuine duplicate.
fn _install_member(
    c: Ctx,
    d: TypeDecl,
    fn_node: Node,
    is_getter: bool,
    is_setter: bool,
    own_from: int
) -> void {
    let name = fn_node.opt_text("name")
    if name is null { return }

    let ty = _member_type_of(c, fn_node, is_getter, is_setter)
    let is_static = fn_node.flag("isStatic")

    let symbol = c.resolution.symbol_of(fn_node)
    if symbol is not null and c.symbol_type(symbol) is null {
        var bound = ty
        if not is_getter and not is_setter { bound = signature_of(c, fn_node, true) }
        c.set_symbol_type(symbol, bound)
    }
    c.set_type(fn_node, ty)

    # `self` is mandatory on an instance method and forbidden on a static one
    # (spec §7.1). Properties take no `self` at all.
    if not is_getter and not is_setter {
        if not is_static and self_skip_of(fn_node) == 0 {
            c.error(
                E0703_MISSING_SELF,
                fn_node.span,
                f"instance method `{name}` must take `self` as its first parameter"
            )
        }
        let params = fn_node.records("params")
        if is_static and params.len() > 0 and _record_name(params[0]) == "self" {
            c.error(
                E0707_STATIC_WITH_SELF,
                fn_node.span,
                f"`static fn {name}` must not take `self`"
            )
        }
    }

    let existing = _own_index(d, name, is_setter, not is_setter)
    if existing >= own_from {
        let diag = c.error(
            E0709_DUPLICATE_MEMBER,
            fn_node.span,
            f"`{name}` is already declared in `{d.name}`"
        )
        diag.label(d.span, "in this declaration")
        return
    }
    if existing == _NOT_FOUND and find_field(d, name) is not null {
        c.error(
            E0709_DUPLICATE_MEMBER,
            fn_node.span,
            f"`{name}` is already declared as a field of `{d.name}`"
        )
        return
    }

    # `init` is not an override: a constructor belongs to its own class and is
    # reached through `super(...)`, never through dynamic dispatch, so a subclass
    # is free to take different arguments.
    if existing == _NOT_FOUND and not is_init_method(fn_node) {
        let overridden = _inherited_pick(d.superclass, name, is_setter, not is_setter)
        if overridden is not null {
            let (ok, why) = _signature_compatible(c, ty, overridden.ty)
            if not ok {
                let sub = render(ty)
                let base = render(overridden.ty)
                let diag = c.error(
                    E0704_INCOMPATIBLE_OVERRIDE,
                    fn_node.span,
                    f"`{name}` overrides an incompatible signature: {why ?? ""}"
                )
                diag.with_note(f"overriding `{base}` with `{sub}`")
                let at = overridden.node
                if at is not null { diag.label(at.span, "the overridden declaration") }
                diag.with_help("parameters may only widen and the return type may only narrow")
            }
        }
    }

    # A trait body declares an interface, so its members are public whether or
    # not `pub` is written: §7.1's "private unless pub" governs classes, and
    # §7.2 makes traits types, through which a private member could never be
    # reached.
    var vis = VIS_PUBLIC
    if not d.is_trait { vis = _node_visibility(fn_node) }

    let info = MemberInfo(
        name,
        ty: _or_any(c, ty),
        visibility: vis,
        is_static: is_static,
        is_getter: is_getter,
        is_setter: is_setter,
        is_abstract: fn_node.child("body") is null,
        node: fn_node
    )
    if existing == _NOT_FOUND { d.members.push(info) } else { d.members[existing] = info }
}

# ------------------------------------------------------------------
# Conformance
# ------------------------------------------------------------------

# `verifyTraits`. An abstract class is exempt, and so is a trait: neither is
# the implementing type.
fn _verify_traits(c: Ctx, d: TypeDecl) -> void {
    if d.is_abstract or d.is_trait { return }

    for (index, trait_decl) in d.traits.enumerate() {
        var binding: list[Type] = []
        if index < d.trait_bindings.len() { binding = d.trait_bindings[index] }

        for requirement in trait_decl.members {
            if not requirement.is_abstract { continue }  # has a default body
            let name = requirement.name
            # The requirement as this implementer wrote it: a generic trait
            # demands its arguments substituted in, not its own parameters.
            let required = _apply_trait_binding(c, d, trait_decl, binding, requirement.ty)
            let provided = _provided_member(c, d, name, requirement.is_setter)

            if provided is null or provided.is_abstract {
                let diag = c.error(
                    E0705_TRAIT_NOT_IMPLEMENTED,
                    d.span,
                    f"`{d.name}` does not implement `{name}` required by trait `{trait_decl.name}`"
                )
                let at = requirement.node
                if at is not null { diag.label(at.span, "required here") }
                let signature = _render_required_method(name, requirement.node, required)
                diag.with_help(f"add `{signature}` to `{d.name}`")
                continue
            }

            let (ok, why) = _signature_compatible(c, provided.ty, required)
            if ok { continue }

            let got = render(provided.ty)
            let want = render(required)
            var span = d.span
            let at = provided.node
            if at is not null { span = at.span }
            let diag = c.error(
                E0705_TRAIT_NOT_IMPLEMENTED,
                span,
                f"`{d.name}` implements `{name}` as `{got}`, but trait " +
                    f"`{trait_decl.name}` requires `{want}`: {why ?? ""}"
            )
            let declared_at = requirement.node
            if declared_at is not null { diag.label(declared_at.span, "required here") }

            # After substitution the requirement no longer reads like the trait
            # body, so say where the substituted form came from.
            if required == requirement.ty { continue }
            let declared = render(requirement.ty)
            # With no trait arguments the only substitution that can have fired
            # is `Self`, and naming the trait again would say nothing.
            if binding.len() == 0 {
                diag.with_note(
                    f"`{name}` is declared `{declared}` in `{trait_decl.name}`, " +
                        f"where `Self` is `{d.name}`"
                )
            } else {
                let written = _render_trait_ref(trait_decl, binding)
                diag.with_note(
                    f"`{name}` is declared `{declared}` in `{trait_decl.name}`, which " +
                        f"`{d.name}` implements as `{written}`"
                )
            }
        }
    }
}

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------

# `resolveDeclReference`. Answers `(decl, opaque_name)`: the declaration when
# this unit can see one, and otherwise the name of a trait or class that
# resolved to a type but to no declaration here.
fn _resolve_decl_reference(
    c: Ctx,
    ref: TypeExpr,
    want_trait: bool,
    self_decl: TypeDecl
) -> (TypeDecl?, str?) {
    let word = if want_trait { "trait" } else { "class" }
    let t = c.resolve_ast_type(c, ref)
    let target = _type_decl_of(t)

    if target is null {
        # A trait or class that reached this module through the prelude or an
        # import has a type but no declaration to lay out here. It is opaque,
        # not wrong: its contract was checked where it was declared and is
        # enforced at run time, so `class P: Printable` must be accepted with
        # only spec §9's re-export in scope.
        var opaque = false
        if t is not null {
            opaque = if want_trait { t.kind == TyKind.Trait } else { t.kind == TyKind.Class }
        }
        if opaque and t is not null { return (null, t.name) }
        if not _is_any(t) {
            c.error(E0700_UNKNOWN_CLASS, ref.span, f"`{ref.name ?? "this type"}` is not a {word}")
        }
        return (null, null)
    }

    if want_trait != target.is_trait or not want_trait and target.is_enum {
        c.error(E0700_UNKNOWN_CLASS, ref.span, f"`{target.name}` is not a {word}")
        return (null, null)
    }

    if target == self_decl or c.layout_status.get(id(target), 0) == 1 {
        let diag = c.error(
            E0706_CYCLIC_INHERITANCE,
            ref.span,
            f"`{target.name}` would inherit from itself"
        )
        diag.label(target.span, "the cycle passes through here")
        return (null, null)
    }

    layout_decl(c, target)
    return (target, null)
}

fn _layout_class(c: Ctx, d: TypeDecl, node: Node) -> void {
    let mark = c.type_name_mark()
    push_generics(c, node.records("generics"))
    push_self(c, d.ty)

    var parent: TypeDecl? = null
    var opaque_parent: str? = null
    let written_super = node.type_field("superclass")
    if written_super is not null {
        let (found, opaque) = _resolve_decl_reference(c, written_super, false, d)
        parent = found
        opaque_parent = opaque
        # A named parent that produced no declaration is a parent this unit
        # cannot enumerate, whatever the reason. Either way the members it might
        # carry are not this unit's to deny.
        if parent is null and opaque_parent is null { opaque_parent = written_super.name }
    }
    d.superclass = parent
    d.opaque_super = opaque_parent

    for ref in node.type_list("traits") {
        let (trait_decl, named_opaque) = _resolve_decl_reference(c, ref, true, d)
        var opaque = named_opaque
        # Same rule as the superclass above, and for the same reason.
        if trait_decl is null and opaque is null { opaque = ref.name }
        if opaque is not null { d.opaque_traits.push(opaque) }
        if trait_decl is null { continue }
        d.trait_bindings.push(_trait_binding_of(c, trait_decl, ref))
        d.traits.push(trait_decl)
    }

    var next_slot = _next_slot(parent)
    for record in node.records("fields") { next_slot = _add_field(c, d, record, next_slot) }

    for method in node.child_list("methods") { _install_member(c, d, method, false, false, 0) }
    for getter in node.child_list("getters") { _install_member(c, d, getter, true, false, 0) }
    for setter in node.child_list("setters") { _install_member(c, d, setter, false, true, 0) }

    # Trait satisfaction is checked before anything can conclude that the class
    # is abstract, or an unimplemented requirement would silently make the class
    # abstract instead of reporting E0705.
    _verify_traits(c, d)
    c.type_name_restore(mark)
}

fn _layout_trait(c: Ctx, d: TypeDecl, node: Node) -> void {
    let mark = c.type_name_mark()
    push_generics(c, node.records("generics"))
    push_self(c, null)

    for ref in node.type_list("supers") {
        let (super_decl, named_opaque) = _resolve_decl_reference(c, ref, true, d)
        var opaque = named_opaque
        if super_decl is null and opaque is null { opaque = ref.name }
        if opaque is not null { d.opaque_traits.push(opaque) }
        if super_decl is null { continue }
        # A null `Self`: a supertrait's `Self` stays abstract here, because this
        # trait is not the implementing type either.
        d.trait_bindings.push(_trait_binding_of(c, super_decl, ref))
        d.traits.push(super_decl)
    }

    # The one place this port copies. An inherited requirement is restated in
    # this trait's own parameters: `trait Sorted[T]: Comparable[T]` requires
    # `compare(T)`, not the supertrait's bare parameter — and `verify_traits`
    # enumerates `trait.members` rather than walking, so a requirement that was
    # not restated would never be checked at all.
    for (index, super_decl) in d.traits.enumerate() {
        var binding: list[Type] = []
        if index < d.trait_bindings.len() { binding = d.trait_bindings[index] }
        for inherited in super_decl.members {
            if _own_index(d, inherited.name, inherited.is_setter, false) != _NOT_FOUND { continue }
            d.members.push(
                MemberInfo(
                    inherited.name,
                    ty: _or_any(c, _apply_trait_binding(c, d, super_decl, binding, inherited.ty)),
                    visibility: inherited.visibility,
                    is_static: inherited.is_static,
                    is_getter: inherited.is_getter,
                    is_setter: inherited.is_setter,
                    is_abstract: inherited.is_abstract,
                    node: inherited.node
                )
            )
        }
    }

    let own_from = d.members.len()
    for method in node.child_list("methods") {
        _install_member(c, d, method, false, false, own_from)
    }
    c.type_name_restore(mark)
}

fn _layout_enum(c: Ctx, d: TypeDecl, node: Node) -> void {
    let u = c.universe
    let mark = c.type_name_mark()
    push_generics(c, node.records("generics"))
    push_self(c, d.ty)

    let own = _decl_type(c, d)
    for record in node.records("variants") {
        let name = _record_name(record)
        if name is null { continue }

        # A unit variant is a value of the enum; a payload variant is the
        # constructor for one. `slot` carries the tag.
        var ty = own
        let params = _record_params(record)
        if params.len() > 0 {
            var payload: list[Type] = []
            for parameter in params {
                payload.push(_or_any(c, c.resolve_ast_type(c, _record_type(parameter))))
            }
            ty = fn_of(u, payload, own, 0)
        }

        if d.find_field(name) is not null {
            c.error(
                E0709_DUPLICATE_MEMBER,
                _record_span(record),
                f"`{name}` is already a variant of `{d.name}`"
            )
            continue
        }
        let index = d.fields.len()
        d.fields.push(FieldInfo(name, ty: ty, slot: index, visibility: VIS_PUBLIC, is_static: true))
    }
    d.variant_count = d.fields.len()

    for method in node.child_list("methods") { _install_member(c, d, method, false, false, 0) }
    c.type_name_restore(mark)
}

#: Lay `decl` out, so that its fields and members can be looked up.
#:
#: `jaiChkLayoutDecl`. Idempotent, and re-entrant on purpose: a superclass or a
#: trait is laid out on demand from the middle of the declaration that names it,
#: and a declaration reached while it is still in progress is the cycle `E0706`
#: reports. Install it as `Ctx.layout_decl`, or let `install` do it.
#:
#: ```
#: from jaithon.ast import class_field
#: let c = Ctx(Universe())
#: install(c)
#: let x = class_field("x", TypeExpr.named("int"))
#: let y = class_field("y", TypeExpr.named("int"))
#: let node = Node(NodeKind.ClassDecl, {"name": "Point", "fields": [x, y]})
#: let point = register_decl(c, node) ?? TypeDecl("Point", node.span)
#: layout_decl(c, point)
#: point.fields.len()       # 2
#: point.fields[1].slot     # 1
#: c.bag.items.len()        # 0
#: ```
pub fn layout_decl(c: Ctx, decl: TypeDecl?) -> void {
    if decl is null { return }
    let key = id(decl)
    if c.layout_status.get(key, 0) != 0 { return }
    c.layout_status[key] = 1

    let node = decl.node
    if node is not null {
        match node.kind {
            NodeKind.ClassDecl => { _layout_class(c, decl, node) }
            NodeKind.TraitDecl => { _layout_trait(c, decl, node) }
            NodeKind.EnumDecl => { _layout_enum(c, decl, node) }
            _ => { }
        }
    }
    c.layout_status[key] = 2
}

# ------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------

#: Whether `fn_node` is a constructor.
#:
#: ```
#: from jaithon.ast import param
#: is_init_method(Node.fn_decl("init", [param("self")], Node.block([])))    # true
#: is_init_method(Node.fn_decl("area", [param("self")], Node.block([])))    # false
#: is_init_method(Node.block([]))                                          # false
#: ```
pub fn is_init_method(fn_node: Node?) -> bool {
    if fn_node is null or fn_node.kind != NodeKind.FnDecl { return false }
    return fn_node.opt_text("name") == "init"
}

#: The return type `fn_node` was declared with.
#:
#: `jaiChkDeclaredReturnType`, with the one place where omission means `void`
#: rather than `any`: a constructor. Answers null only when a written annotation
#: failed to resolve, which is what `Ctx.resolve_ast_type`'s default always does.
#:
#: ```
#: from jaithon.ast import param
#: let c = Ctx(Universe())
#: let init = Node.fn_decl("init", [param("self")], Node.block([]))
#: declared_return_type(c, init, true) == c.universe.t_void     # true
#: declared_return_type(c, init, false) == c.universe.t_any     # true
#: ```
pub fn declared_return_type(c: Ctx, fn_node: Node?, is_method: bool) -> Type? {
    if fn_node is null { return c.universe.t_any }
    let written = fn_node.type_field("returnType")
    if written is not null and written.kind != TypeKind.Infer {
        return c.resolve_ast_type(c, written)
    }
    if is_method and is_init_method(fn_node) { return c.universe.t_void }
    return c.universe.t_any
}

#: The `Fn` type of `fn_node`, dropping the receiver when `is_method`.
#:
#: `jaiChkFunctionType`. The generic parameters are in scope while the
#: parameters and the return type resolve, and the variadic and kwrest flags
#: come off the parameter list — the type carries the flags, and the *binding*
#: types those two parameters have inside the body are `check_body`'s business.
#:
#: A parameter whose written type does not resolve reads as `any`, where the C
#: leaves a null in the array: `fn_of` takes a `list[Type]`, and `any` is what
#: every consumer treats a null parameter type as anyway.
#:
#: ```
#: from jaithon.ast import param
#: let c = Ctx(Universe())
#: let method = Node.fn_decl("scale", [param("self"), param("by")], Node.block([]))
#: render(signature_of(c, method, true))     # "fn(any) -> any"
#: render(signature_of(c, method, false))    # "fn(any, any) -> any"
#: ```
pub fn signature_of(c: Ctx, fn_node: Node?, is_method: bool) -> Type {
    let u = c.universe
    if fn_node is null { return u.t_any }

    let mark = c.type_name_mark()
    push_generics(c, fn_node.records("generics"))

    let skip = if is_method { self_skip_of(fn_node) } else { 0 }
    let params = fn_node.records("params")
    var types: list[Type] = []
    var flags = 0
    var index = skip
    while index < params.len() {
        let record = params[index]
        if _record_flag(record, "isVariadic") { flags = flags | FN_VARIADIC }
        if _record_flag(record, "isKwRest") { flags = flags | FN_KWREST }
        types.push(_or_any(c, c.resolve_ast_type(c, _record_type(record))))
        index += 1
    }

    let returns = declared_return_type(c, fn_node, is_method)
    c.type_name_restore(mark)
    return fn_of(u, types, returns, flags)
}

#: `signature_of` with `is_method` false — the shape `Ctx.function_type` takes.
#:
#: ```
#: from jaithon.ast import param
#: let c = Ctx(Universe())
#: render(function_type(c, Node.fn_decl("f", [param("n")], Node.block([]))))   # "fn(any) -> any"
#: render(function_type(c, null))                                             # "any"
#: ```
pub fn function_type(c: Ctx, fn_node: Node?) -> Type {
    return signature_of(c, fn_node, false)
}

# The type a parameter's binding has inside the body, which is not the type in
# the signature for `...xs` and `**opts`.
fn _parameter_binding_type(c: Ctx, record: dict[str, any]) -> Type {
    let declared = _or_any(c, c.resolve_ast_type(c, _record_type(record)))
    if _record_flag(record, "isVariadic") { return list_of(c.universe, declared) }
    if _record_flag(record, "isKwRest") { return dict_of(c.universe, c.universe.t_str, declared) }
    return declared
}

#: Check the body of `fn_node`, and its parameters' defaults.
#:
#: `jaiChkFunction`. `owner` is the declaration a method belongs to, which
#: becomes `Ctx.current_class` for the body and the type its `self` binds to;
#: `is_method` says whether the first parameter is that receiver.
#:
#: The block itself goes to `Ctx.check_block`, whose default does nothing, so
#: today this checks the signature, the parameter defaults and — for an
#: expression-bodied function — the body expression. E0601 is reported here
#: because it is a fact about the declaration rather than about any statement.
#:
#: ```
#: from jaithon.ast import param
#: let c = Ctx(Universe())
#: let f = Node.fn_decl("f", [param("n", null, Node.int_lit(1))], Node.block([]))
#: check_body(c, f, null, false)
#: c.bag.items.len()          # 0
#: c.current_return_type is null   # true, restored on the way out
#: ```
pub fn check_body(c: Ctx, fn_node: Node?, owner: TypeDecl?, is_method: bool) -> void {
    if fn_node is null { return }
    let u = c.universe

    let name_mark = c.type_name_mark()
    push_generics(c, fn_node.records("generics"))

    let narrow_mark = c.narrow_mark()
    let saved_return = c.current_return_type
    let saved_class = c.current_class
    if owner is not null { c.current_class = owner }

    let skip = if is_method { self_skip_of(fn_node) } else { 0 }
    for (index, record) in fn_node.records("params").enumerate() {
        var bound = u.t_any
        if index < skip {
            bound = _decl_type(c, owner)
        } else {
            bound = _parameter_binding_type(c, record)
            if c.strict and _record_type(record) is null and not _record_flag(record, "isKwRest") {
                let shown = _record_name(record) ?? "?"
                let diag = c.error(
                    E0400_TYPE_MISMATCH,
                    _record_span(record),
                    f"parameter `{shown}` has no type annotation"
                )
                diag.with_help("--strict requires every parameter to be annotated")
            }
        }
        c.set_symbol_type(c.resolution.symbol_of(record), bound)

        let default_value = _record_default(record)
        if default_value is null { continue }
        c.apply_context(c, default_value, bound)
        let got = value(c, default_value)
        let _ok = c.require_assignable(
            c,
            default_value,
            got,
            bound,
            E0400_TYPE_MISMATCH,
            "mismatched type for a default value"
        )
    }

    var ret = declared_return_type(c, fn_node, is_method)
    c.current_return_type = ret

    let written = fn_node.type_field("returnType")
    let annotated = written is not null and written.kind != TypeKind.Infer
    let body = fn_node.child("body")
    if body is not null {
        if body.kind == NodeKind.Block {
            c.check_block(c, body)
        } else {
            # A lambda's body is its return value.
            if not _is_any(ret) { c.apply_context(c, body, ret) }
            let got = value(c, body)
            if annotated {
                let _ok = c.require_assignable(
                    c,
                    body,
                    got,
                    ret,
                    E0408_RETURN_TYPE_MISMATCH,
                    "mismatched return type"
                )
            } else {
                ret = got
            }
        }
    }

    _check_falls_off_end(c, fn_node, body, ret, annotated)

    c.current_return_type = saved_return
    c.current_class = saved_class
    c.narrow_restore(narrow_mark)
    c.type_name_restore(name_mark)
}

# E0601: an annotated non-void function must not fall off the end. A generator
# returns through `yield`, so it is exempt.
fn _check_falls_off_end(c: Ctx, fn_node: Node, body: Node?, ret: Type?, annotated: bool) -> void {
    if body is null or body.kind != NodeKind.Block { return }
    if fn_node.flag("isGenerator") or not annotated { return }
    if _is_void(c, ret) or _is_any(ret) or _is_never(ret) { return }
    if ret is null or ret.kind == TyKind.Null or is_optional(ret) { return }
    if always_diverges(body) { return }

    let written = fn_node.type_field("returnType")
    if written is null { return }

    let want = render(ret)
    let shown = fn_node.opt_text("name") ?? "this function"
    let diag = c.error(
        E0601_MISSING_RETURN,
        fn_node.span,
        f"`{shown}` must return `{want}` on every path"
    )
    diag.label(written.span, f"declared to return `{want}`")
    diag.with_help("add a `return` at the end, or make the last branch an `else`")
}

#: `check_body` with no owner and no receiver — the shape `Ctx.check_function`
#: takes, which is how a lambda and an anonymous function reach it.
pub fn check_function(c: Ctx, fn_node: Node?) -> void {
    check_body(c, fn_node, null, false)
}

# ------------------------------------------------------------------
# Bodies
# ------------------------------------------------------------------

# `super(...)` must open the constructor when the parent declares one, so that a
# subclass can never observe a half-initialised parent (spec §7.1).
fn _check_super_init_position(c: Ctx, d: TypeDecl, init: Node) -> void {
    let parent = d.superclass
    if parent is null { return }
    let (parent_init, _owner) = find_init_decl(parent)
    if parent_init is null { return }

    let body = init.child("body")
    if body is null or body.kind != NodeKind.Block { return }

    var first_super: Node? = null
    for candidate in preorder(body) {
        if candidate.kind != NodeKind.Call { continue }
        let callee = candidate.child("callee")
        if callee is null or callee.kind != NodeKind.Super { continue }
        first_super = candidate
        break
    }
    if first_super is null { return }

    let stmts = body.child_list("stmts")
    if stmts.len() > 0 and stmts[0].kind == NodeKind.ExprStmt {
        let held = stmts[0].child("expr")
        if held is not null and id(held) == id(first_super) { return }
    }

    let diag = c.error(
        E0708_SUPER_INIT_NOT_FIRST,
        first_super.span,
        "`super(...)` must be the first statement of `init`"
    )
    diag.label(parent.span, f"`{parent.name}` declares an `init`")
    diag.with_help("move the call to the top of `init`")
}

fn _check_class_body(c: Ctx, node: Node) -> void {
    # `decl_for_node` is the C's `jaiTypeDeclFind(name)` followed by
    # `d->decl != node`: a declaration whose name another one already held was
    # never registered against this node, so it answers null and is skipped.
    let d = c.decl_for_node(node)
    if d is null { return }
    layout_decl(c, d)

    let mark = c.type_name_mark()
    push_generics(c, node.records("generics"))
    push_self(c, _decl_type(c, d))
    let saved_class = c.current_class
    c.current_class = d

    for record in node.records("fields") {
        let default_value = _record_default(record)
        if default_value is null { continue }
        let want = c.resolve_ast_type(c, _record_type(record))
        c.apply_context(c, default_value, want)
        let got = value(c, default_value)
        let _ok = c.require_assignable(
            c,
            default_value,
            got,
            want,
            E0400_TYPE_MISMATCH,
            "mismatched type for a field default"
        )
    }

    for method in node.child_list("methods") {
        check_body(c, method, d, true)
        if is_init_method(method) { _check_super_init_position(c, d, method) }
    }
    for getter in node.child_list("getters") { check_body(c, getter, d, true) }
    for setter in node.child_list("setters") { check_body(c, setter, d, true) }

    c.current_class = saved_class
    c.type_name_restore(mark)
}

fn _check_trait_body(c: Ctx, node: Node) -> void {
    let d = c.decl_for_node(node)
    if d is null { return }
    layout_decl(c, d)

    let mark = c.type_name_mark()
    push_generics(c, node.records("generics"))
    push_self(c, null)
    let saved_class = c.current_class
    c.current_class = d

    for method in node.child_list("methods") {
        if method.child("body") is null { continue }  # a requirement, not a body
        check_body(c, method, d, true)
    }

    c.current_class = saved_class
    c.type_name_restore(mark)
}

fn _check_enum_body(c: Ctx, node: Node) -> void {
    let d = c.decl_for_node(node)
    if d is null { return }
    layout_decl(c, d)

    let mark = c.type_name_mark()
    push_generics(c, node.records("generics"))
    push_self(c, _decl_type(c, d))
    let saved_class = c.current_class
    c.current_class = d

    for record in node.records("variants") {
        for parameter in _record_params(record) {
            let default_value = _record_default(parameter)
            if default_value is null { continue }
            let _got = value(c, default_value)
        }
    }
    for method in node.child_list("methods") { check_body(c, method, d, true) }

    c.current_class = saved_class
    c.type_name_restore(mark)
}

#: Check one declaration: a function, a class, a trait, an enum, a type alias or
#: an import.
#:
#: `jaiChkDecl`, and the shape `Ctx.check_decl` takes. A `let`, `var` or `const`
#: is **not** one of these — `jaithon.ast.is_declaration` puts it with the
#: statements, exactly as the C does — so it has its own entry point in
#: `check_var_decl` below, which `stmt.jai` will route to.
#:
#: ```
#: from jaithon.ast import param
#: let c = Ctx(Universe())
#: install(c)
#: let f = Node.fn_decl("twice", [param("n")], Node.block([]))
#: check_decl(c, f)
#: render(c.type_of(f))       # "fn(any) -> any"
#: ```
pub fn check_decl(c: Ctx, node: Node?) -> void {
    if node is null { return }
    match node.kind {
        NodeKind.FnDecl => {
            let symbol = c.resolution.symbol_of(node)
            if symbol is not null and c.symbol_type(symbol) is null {
                c.set_symbol_type(symbol, signature_of(c, node, false))
            }
            if c.type_of(node) is null {
                var known: Type? = null
                if symbol is not null { known = c.symbol_type(symbol) }
                c.set_type(node, known ?? signature_of(c, node, false))
            }
            check_body(c, node, null, false)
        }

        NodeKind.ClassDecl => { _check_class_body(c, node) }
        NodeKind.TraitDecl => { _check_trait_body(c, node) }
        NodeKind.EnumDecl => { _check_enum_body(c, node) }

        NodeKind.TypeDecl => {
            let symbol = c.resolution.symbol_of(node)
            if symbol is not null and c.symbol_type(symbol) is null {
                c.set_symbol_type(symbol, c.resolve_ast_type(c, node.type_field("aliased")))
            }
        }

        NodeKind.Import => {
            let symbol = c.resolution.symbol_of(node)
            if symbol is not null and c.symbol_type(symbol) is null {
                c.set_symbol_type(symbol, named(c.universe, TyKind.Module, node.opt_text("path")))
            }
        }

        _ => { }
    }
}

# ------------------------------------------------------------------
# Value declarations
# ------------------------------------------------------------------

# An unannotated `var` initialised to null would otherwise be pinned to the null
# type and reject every later assignment; the binding is dynamic.
fn _widen_inferred(u: Universe, kind: DeclKind, t: Type?) -> Type {
    if t is null { return u.t_any }
    if kind == DeclKind.Var and (t.kind == TyKind.Null or t.kind == TyKind.Never) { return u.t_any }
    if t.kind == TyKind.Never { return u.t_any }
    return t
}

# The symbol a `let`, `var` or `const` binds: the declaration's own, or the one
# its `PatBind` leaf carries.
fn _decl_symbol(c: Ctx, node: Node) -> Symbol? {
    let own = c.resolution.symbol_of(node)
    if own is not null { return own }
    let pattern = node.child("pattern")
    if pattern is null or pattern.kind != NodeKind.PatBind { return null }
    return c.resolution.symbol_of(pattern)
}

# `foldConstDecl`. The folded value is kept so every use can inline it — on the
# C's `Symbol.constValue`, and here in `Ctx.consts` under the binding's name,
# which is where `fold.jai` and `expr.jai` look for it.
fn _fold_const_decl(c: Ctx, node: Node, symbol: Symbol?) -> void {
    let init = node.child("init")
    if init is null {
        c.error(E0304_MISSING_INITIALIZER, node.span, "a `const` declaration needs an initialiser")
        return
    }

    let before = c.bag.error_count
    let folded = const_eval(init, c.bag, c.consts)
    if folded.is_none() {
        if c.bag.error_count == before {
            let diag = c.error(
                E0303_NON_CONSTANT_CONST,
                init.span,
                "the initialiser of a `const` must be a constant expression"
            )
            diag.with_help("use `let` if the value is only known at run time")
        }
        return
    }

    let literal = Node(NodeKind.NullLit, null, init.span)
    let _kept = const_replace(c.universe, literal, folded, c.dump)
    let _inlined = const_replace(c.universe, init, folded, c.dump)
    if symbol is not null { c.consts[symbol.name] = literal }
}

#: Check a `let`, `var` or `const` and give its binding a type.
#:
#: `checkVarDecl` in `src/sema/check_stmt.c`. It lives in this file because it
#: is a declaration and because `stmt.jai` does not exist yet; when it does, its
#: `AST_VAR_DECL` case calls this rather than reimplementing it.
#:
#: An annotation is the binding's type and the initialiser is checked against
#: it; without one the initialiser's own type is inferred, widened to `any` for
#: a `var` bound to null so that a later assignment is not refused.
#:
#: ```
#: let c = Ctx(Universe())
#: let node = Node.var_decl(DeclKind.Var, Node.bind("x"), null, Node.null_lit())
#: check_var_decl(c, node)
#: render(c.type_of(node))    # "any"
#: let fixed = Node.var_decl(DeclKind.Let, Node.bind("n"), null, Node.int_lit(1))
#: check_var_decl(c, fixed)
#: render(c.type_of(fixed))   # "int"
#: ```
pub fn check_var_decl(c: Ctx, node: Node?) -> void {
    if node is null { return }
    let u = c.universe

    let annotation = node.type_field("declaredType")
    var declared: Type? = null
    if annotation is not null and annotation.kind != TypeKind.Infer {
        declared = c.resolve_ast_type(c, annotation)
    }

    let kind: DeclKind = node.get("varDeclKind")
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
            bound = _widen_inferred(u, kind, got)
        }
    }

    let resolved = bound ?? u.t_any
    c.set_type(node, resolved)

    let symbol = _decl_symbol(c, node)
    c.set_symbol_type(symbol, resolved)
    let pattern = node.child("pattern")
    if pattern is not null { c.check_pattern(c, pattern, resolved) }

    if kind == DeclKind.Const { _fold_const_decl(c, node, symbol) }
}

# ------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------

#: Install this file's four callbacks over `Ctx`'s defaults.
#:
#: `check_internal.h` is the C's whole coupling between the five checker files;
#: this is one direction of it. `checker.jai` will call this rather than
#: assigning each field, so that a callback added here reaches every construction
#: site at once.
#:
#: ```
#: let c = Ctx(Universe())
#: let node = Node(NodeKind.ClassDecl, {"name": "Box"})
#: let box = register_decl(c, node) ?? TypeDecl("Box", node.span)
#: c.layout_decl(c, box)
#: c.layout_status.len()    # 0, `Ctx`'s own default lays nothing out
#: install(c)
#: c.layout_decl(c, box)
#: c.layout_status.len()    # 1
#: ```
pub fn install(c: Ctx) -> void {
    c.layout_decl = layout_decl
    c.check_decl = check_decl
    c.check_function = check_function
    c.function_type = function_type
}
