#: Statements, control flow, and the narrowing a branch is allowed to keep.
#:
#: A port of `src/sema/check_stmt.c`: patterns and `match`, the `let`/`var`/
#: `const` that appear inside a body, assignment, blocks and loops, `return`,
#: `throw`, `try`, `defer`, `assert`, and function bodies.
#:
#: **There is no truthiness** (spec §5.1), so every condition goes through
#: `expr.condition` and is checked for `bool` rather than coerced. And
#: **narrowing is scoped**: a fact learned by `if x is C` is pushed on entry to
#: the arm it holds in and popped on the way out, which is why so many functions
#: here are a `narrow_mark`/`narrow_restore` pair around a recursive call.
#:
#: Three narrowing rules are worth stating outright:
#:
#: - **The false edge is collected, not inverted.** `a or b` proves nothing when
#:   true, while `not (a or b)` proves both parts when false, and the second is
#:   the `if x is null or x <= 0 { return }` guard the library is written with.
#: - **A diverging `then` with no `else` narrows the rest of the block.** Those
#:   facts are deliberately not restored; the enclosing `block` unwinds them.
#: - **A store ends a fact.** The store is checked against the declaration —
#:   `Ctx.declared_type` — and then calls `Ctx.narrow_invalidate`.
#:
#: `Ctx` is the seam and `expr.jai` owns it; `install` fills in five of its
#: callbacks. `checker.jai` calls `install` too and then replaces
#: `Ctx.check_decl`, `Ctx.resolve_ast_type` and `Ctx.layout_decl`, none of which
#: this file can supply. Until it lands, every written annotation here reads as
#: `any`, a declaration inside a block is unchecked, and a class is never laid
#: out.
#:
#: Two gaps are the AST's rather than a callback's: a `for` loop's hidden
#: iterator binding is never typed, because the Jaithon `For` schema has no such
#: field; and a `const`'s folded value lives in `Ctx.consts` keyed by **name**
#: rather than on the `Symbol`, so two `const K` in different scopes collide.
from jaithon.ast import DeclKind, Node, NodeKind, Span, TypeExpr, TypeKind
from jaithon.ast import always_diverges, is_declaration, is_expression, is_pattern, op_text
from jaithon.compile.check.assign import assignable
from jaithon.compile.check.ctx import Ctx, E0411_INT_FLOAT_MIX, subject_of
from jaithon.compile.check.expr import E0400_TYPE_MISMATCH, E0404_NOT_INDEXABLE
from jaithon.compile.check.expr import E0406_BAD_OPERAND_TYPES, E0410_UNKNOWN_MEMBER
from jaithon.compile.check.expr import E0701_PRIVATE_ACCESS, W0102_UNREACHABLE_CODE, condition
from jaithon.compile.check.expr import expr, iterable_element, self_skip_of, slice
from jaithon.compile.check.expr import static_target_decl, value
from jaithon.compile.check.expr import visibility_word, visible_from
from jaithon.compile.check.fold import ConstKind, ConstValue, const_eval, const_replace
from jaithon.compile.check.kinds import MAX_DECL_DEPTH, TyKind
from jaithon.compile.check.nominal import find_accessor, find_member, owner_of_field
from jaithon.compile.check.operator import binary_result, widens_operands
from jaithon.compile.check.predicate import is_indexable, is_optional
from jaithon.compile.check.relate import join
from jaithon.compile.check.render import render
from jaithon.compile.check.suggest import suggest_member
from jaithon.compile.check.ty import FieldInfo, Type, TypeDecl
from jaithon.compile.check.union import optional_of, union_of
from jaithon.compile.check.universe import dict_of, fn_of, generic_param, list_of
from jaithon.compile.symbol import Symbol

#: A store into something declared `let`.
#:
#: The codes below are the `JaiDiagCode` names in `src/common/diag.h`. Both the
#: code and the message text are a contract: Phase 4 turns `tests/errors` into
#: goldens and a reworded message fails there.
pub const E0301_ASSIGN_TO_IMMUTABLE: str = "E0301"

#: A `const` whose initialiser has no compile-time value.
pub const E0303_NON_CONSTANT_CONST: str = "E0303"

#: A `const` with no initialiser at all.
pub const E0304_MISSING_INITIALIZER: str = "E0304"

#: A pattern naming a type this unit cannot find.
pub const E0402_UNKNOWN_TYPE: str = "E0402"

#: A `return` that does not fit the enclosing function's declared result.
pub const E0408_RETURN_TYPE_MISMATCH: str = "E0408"

#: A match arm an earlier arm already takes.
pub const E0500_UNREACHABLE_ARM: str = "E0500"

#: A `match` that leaves a value of the subject's type unmatched.
pub const E0501_NON_EXHAUSTIVE_MATCH: str = "E0501"

#: A pattern binding a different number of elements than the subject has.
pub const E0502_PATTERN_ARITY: str = "E0502"

#: One name bound twice by one pattern.
pub const E0503_DUPLICATE_BINDING_IN_PATTERN: str = "E0503"

#: An annotated function that can fall off its end.
pub const E0601_MISSING_RETURN: str = "E0601"

#: A store into a field the class does not declare.
pub const E0702_UNDECLARED_FIELD: str = "E0702"

#: How many names one pattern may bind before duplicates stop being reported.
#:
#: `MAX_BINDS` in `src/sema/check_stmt.c`. The C's caps are stack array sizes,
#: but every one of them is observable: past the cap the C stops reporting, so a
#: port that kept going would produce diagnostics the C does not.
pub const MAX_BINDS: int = 64

#: How many distinct literal arms a `match` tracks for `E0500`.
pub const MAX_TRACKED_LITERALS: int = 64

#: How many enum variants a `match` tracks for `E0500` and `E0501`.
pub const MAX_TRACKED_VARIANTS: int = 256

#: How many sequence lengths a `match` tracks, the width of the C's bitset.
pub const MAX_TRACKED_LENGTHS: int = 64

# `jaiChkIsAny` and the five below it are file-private in `expr.jai` and so
# cannot be imported. Duplicating two lines is cheaper than widening that file's
# surface for one caller.
fn _is_any(t: Type?) -> bool {
    if t is null { return true }
    return t.kind == TyKind.Any or t.kind == TyKind.GenericParam
}

fn _is_never(t: Type?) -> bool {
    return t is not null and t.kind == TyKind.Never
}

# `void` is `null` under the name a return position uses, so this is always
# false in both front ends. It starts reporting the day either universe gives
# `void` an identity of its own.
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

fn _type_decl_of(t: Type?) -> TypeDecl? {
    if t is null { return null }
    if not t.is_named() { return null }
    return t.decl
}

fn _span_of(node: Node?) -> Span {
    if node is null { return Span.none() }
    return node.span
}

# `d->fields` as the C holds it: a parent's entries first, then the
# declaration's own, which is the order `FieldInfo.slot` numbers.
# `layoutClassFields` in `src/sema/check_decl.c` writes that order into the
# child; nothing copies into a Jaithon `TypeDecl`, so the chain is walked here.
# Capped at `MAX_DECL_DEPTH` because this is also asked about declarations still
# being laid out, whose cycles layout has not rejected yet.
fn _layout_fields(decl: TypeDecl?) -> list[FieldInfo] {
    return _collect_fields(decl, 0)
}

fn _collect_fields(decl: TypeDecl?, depth: int) -> list[FieldInfo] {
    if decl is null or depth > MAX_DECL_DEPTH { return [] }
    var out = _collect_fields(decl.superclass, depth + 1)
    for field in decl.fields { out.push(field) }
    return out
}

# The index and not just the entry, because an enum's variants are its first
# `variant_count` fields and that is the only thing distinguishing a variant
# from a static member.
fn _field_index(decl: TypeDecl?, name: str?) -> int {
    if decl is null or name is null { return -1 }
    for (position, field) in _layout_fields(decl).enumerate() {
        if field.name == name { return position }
    }
    return -1
}

fn _sub_pattern(pat: Node, at: int) -> Node? {
    let subs = pat.child_list("subPatterns")
    if at < 0 or at >= subs.len() { return null }
    return subs[at]
}

fn _field_name(pat: Node, at: int) -> str? {
    let names: list[str?] = pat.get("fieldNames")
    if at < 0 or at >= names.len() { return null }
    return names[at]
}

# The C's `BindSet`. Each alternative of an `or` pattern gets a fresh one: the
# alternatives bind the same names by design, and colliding them would report
# every `A(x) | B(x)` as a duplicate.
class _BindSet {
    #: The names bound so far, capped at `MAX_BINDS`.
    pub var names: list[str]
    #: Where each was bound, so the second one can point at the first.
    pub var spans: list[Span]

    #: An empty set.
    pub fn init(self) {
        self.names = []
        self.spans = []
    }
}

fn _record_binding(c: Ctx, binds: _BindSet?, name: str?, span: Span) -> void {
    if binds is null or name is null { return }
    # The C tests only the first byte, so every `_`-prefixed name discards.
    if name.starts_with("_") { return }
    for (position, bound) in binds.names.enumerate() {
        if bound == name {
            let d = c.error(
                E0503_DUPLICATE_BINDING_IN_PATTERN,
                span,
                f"`{name}` is bound twice in the same pattern"
            )
            d.label(binds.spans[position], "first bound here")
            return
        }
    }
    if binds.names.len() >= MAX_BINDS { return }
    binds.names.push(name)
    binds.spans.push(span)
}

# The element type of a sequence subject; a tuple answers per position, and
# `index` of -1 asks for the whole of it.
fn _sequence_element(c: Ctx, subject: Type?, index: int) -> Type {
    let u = c.universe
    if subject is null { return u.t_any }
    match subject.kind {
        TyKind.List | TyKind.Set => {
            return if subject.arg_count() == 1 { subject.args[0] } else { u.t_any }
        }
        TyKind.Tuple => {
            if index >= 0 and index < subject.arg_count() { return subject.args[index] }
            return u.t_any
        }
        _ => { return u.t_any }
    }
}

fn _pattern_sequence(c: Ctx, pat: Node, subject: Type?, binds: _BindSet?) -> void {
    let elems = pat.child_list("elems")
    let count = elems.len()
    let rest = pat.number("restIndex")

    if subject is not null and
        subject.kind == TyKind.Tuple and
        rest < 0 and
        count != subject.arg_count() {
        let got = render(subject)
        let plural = if count == 1 { "" } else { "s" }
        let arity = subject.arg_count()
        c.error(
            E0502_PATTERN_ARITY,
            pat.span,
            f"this pattern has {count} element{plural} but `{got}` has {arity}"
        )
    }

    for (position, sub) in elems.enumerate() {
        var elem: Type = c.universe.t_any
        if position == rest {
            elem = list_of(c.universe, _sequence_element(c, subject, -1))
        } else {
            # Everything past the rest has an unknowable position in a tuple.
            let index = if rest >= 0 and position > rest { -1 } else { position }
            elem = _sequence_element(c, subject, index)
        }
        _pattern_walk(c, sub, elem, binds)
    }
}

fn _pattern_enum(c: Ctx, pat: Node, subject: Type?, binds: _BindSet?) -> void {
    let u = c.universe
    let type_name = pat.opt_text("typeName")
    let variant_name = pat.opt_text("variantName")
    let count = pat.child_list("subPatterns").len()

    var decl = c.decl_named(type_name)
    if decl is null {
        # `match colour { Colour.Red => ... }` names a type the subject already
        # supplies, whether or not the registry has it under that name.
        let from_subject = _type_decl_of(subject)
        if from_subject is not null and from_subject.name == type_name { decl = from_subject }
    }
    if decl is null {
        if not _is_any(subject) {
            let shown = type_name ?? "?"
            c.error(E0402_UNKNOWN_TYPE, pat.span, f"unknown type `{shown}` in pattern")
        }
        var at = 0
        while at < count {
            _pattern_walk(c, _sub_pattern(pat, at), u.t_any, binds)
            at += 1
        }
        return
    }
    c.layout_decl(c, decl)

    let vi = _field_index(decl, variant_name)
    if vi < 0 or vi >= decl.variant_count {
        let shown = variant_name ?? "?"
        let d = c.error(E0410_UNKNOWN_MEMBER, pat.span, f"`{decl.name}` has no variant `{shown}`")
        suggest_member(d, decl, variant_name ?? "")
        var at = 0
        while at < count {
            _pattern_walk(c, _sub_pattern(pat, at), u.t_any, binds)
            at += 1
        }
        return
    }

    # A variant's payload is stored as the `Fn` that builds it, so its
    # parameters are the fields the pattern destructures.
    let payload = _layout_fields(decl)[vi].ty
    let is_fn = payload is not null and payload.kind == TyKind.Fn
    let arity = if is_fn { payload.arg_count() } else { 0 }
    if count != arity {
        let plural = if arity == 1 { "" } else { "s" }
        let shown = variant_name ?? "?"
        let d = c.error(
            E0502_PATTERN_ARITY,
            pat.span,
            f"`{decl.name}.{shown}` takes {arity} field{plural} but the pattern binds {count}"
        )
        d.label(decl.span, f"`{decl.name}` is declared here")
    }
    var at = 0
    while at < count {
        var field: Type = u.t_any
        if is_fn and at < payload.arg_count() { field = payload.args[at] }
        _pattern_walk(c, _sub_pattern(pat, at), field, binds)
        at += 1
    }
}

fn _pattern_class(c: Ctx, pat: Node, subject: Type?, binds: _BindSet?) -> void {
    if pat.opt_text("variantName") is not null {
        _pattern_enum(c, pat, subject, binds)
        return
    }

    let u = c.universe
    let type_name = pat.opt_text("typeName")
    let count = pat.child_list("subPatterns").len()
    let decl = c.decl_named(type_name)
    if decl is null {
        if not _is_any(subject) {
            let shown = type_name ?? "?"
            c.error(E0402_UNKNOWN_TYPE, pat.span, f"unknown type `{shown}` in pattern")
        }
        var missing = 0
        while missing < count {
            _pattern_walk(c, _sub_pattern(pat, missing), u.t_any, binds)
            missing += 1
        }
        return
    }
    c.layout_decl(c, decl)

    # An enum written without a variant is a whole-enum pattern.
    if decl.is_enum and count == 0 { return }

    let fields = _layout_fields(decl)
    if count > fields.len() {
        let plural = if fields.len() == 1 { "" } else { "s" }
        let declared = fields.len()
        c.error(
            E0502_PATTERN_ARITY,
            pat.span,
            f"`{decl.name}` has {declared} field{plural} but the pattern binds {count}"
        )
    }

    var at = 0
    while at < count {
        let field_name = _field_name(pat, at)
        var field_type: Type = u.t_any
        var fi = -1
        if field_name is not null {
            fi = _field_index(decl, field_name)
        } elif at < fields.len() {
            fi = at
        }

        if field_name is not null and fi < 0 {
            let d = c.error(
                E0410_UNKNOWN_MEMBER,
                pat.span,
                f"`{decl.name}` has no field `{field_name}`"
            )
            suggest_member(d, decl, field_name)
        } elif fi >= 0 {
            let field = fields[fi]
            field_type = _or_any(c, field.ty)
            let owner = owner_of_field(decl, field.name)
            if not visible_from(c, owner, field.visibility) {
                let word = visibility_word(field.visibility)
                let owner_name = if owner is null { decl.name } else { owner.name }
                c.error(
                    E0701_PRIVATE_ACCESS,
                    pat.span,
                    f"`{field.name}` is {word} to `{owner_name}`"
                )
            }
        }

        let sub = _sub_pattern(pat, at)
        if sub is not null {
            _pattern_walk(c, sub, field_type, binds)
        } elif field_name is not null {
            _record_binding(c, binds, field_name, pat.span)
        }
        at += 1
    }
}

fn _pattern_walk(c: Ctx, pat: Node?, raw_subject: Type?, binds: _BindSet?) -> void {
    if pat is null { return }
    let u = c.universe
    let subject = _or_any(c, raw_subject)

    match pat.kind {
        NodeKind.PatWildcard => { }

        NodeKind.PatBind => {
            let written = pat.type_field("type")
            var bound: Type? = subject
            if written is not null { bound = c.resolve_ast_type(c, written) }
            _record_binding(c, binds, pat.opt_text("name"), pat.span)
            c.set_symbol_type(c.resolution.symbol_of(pat), _or_any(c, bound))
            c.set_type(pat, _or_any(c, bound))
            return
        }

        NodeKind.PatLiteral => {
            let literal_node = pat.child("value")
            if literal_node is not null {
                c.apply_context(c, literal_node, subject)
                let lit = value(c, literal_node)
                let (forwards, _fg) = assignable(u, lit, subject)
                let (backwards, _bg) = assignable(u, subject, lit)
                if not _is_any(subject) and not _is_any(lit) and not forwards and not backwards {
                    let want = render(subject)
                    let got = render(lit)
                    c.error(
                        E0400_TYPE_MISMATCH,
                        pat.span,
                        f"this pattern can never match: expected `{want}`, found `{got}`"
                    )
                }
            }
        }

        NodeKind.PatRange => {
            for name in ["lo", "hi"] {
                let bound = pat.child(name)
                if bound is null { continue }
                c.apply_context(c, bound, subject)
                value(c, bound)
            }
        }

        NodeKind.PatTuple | NodeKind.PatList => { _pattern_sequence(c, pat, subject, binds) }

        NodeKind.PatClass | NodeKind.PatEnum => { _pattern_class(c, pat, subject, binds) }

        NodeKind.PatOr => {
            for alternative in pat.child_list("elems") {
                _pattern_walk(c, alternative, subject, _BindSet())
            }
        }

        _ => {
            # A destructuring target reuses expression nodes, which keep the
            # type the expression checker gave them.
            if is_expression(pat.kind) {
                expr(c, pat)
                return
            }
        }
    }

    c.set_type(pat, subject)
}

#: Bind `pat` against the type it destructures, reporting what does not fit.
#:
#: `jaiChkPattern`. The shape of the installed `Ctx.check_pattern`, so a
#: comprehension's loop variable is bound through exactly this.
pub fn pattern(c: Ctx, pat: Node?, subject: Type) -> void {
    _pattern_walk(c, pat, subject, _BindSet())
}

# The C's `Coverage`, with its two fixed-size arrays replaced by sets and the
# caps kept as explicit tests: the caps are observable and the sizes are not.
class _Coverage {
    #: Whether some arm binds unconditionally.
    pub var saw_catch_all: bool
    #: Whether a `true` arm was seen, for a `bool` subject.
    pub var covered_true: bool
    #: Whether a `false` arm was seen.
    pub var covered_false: bool
    #: The indices of the enum variants an arm has taken.
    pub var variants: set[int]
    #: The literal values an arm has taken, capped at `MAX_TRACKED_LITERALS`.
    pub var literals: list[ConstValue]
    #: Sequence arms are tracked by the lengths they accept: one entry per exact
    #: length, plus `covered_from` below. A total match does not yield `T?`.
    pub var covered_lengths: set[int]
    #: The shortest length from which a `[..., ...rest]` arm takes everything;
    #: `MAX_TRACKED_LENGTHS` means no open arm was seen.
    pub var covered_from: int

    #: Nothing covered yet.
    pub fn init(self) {
        self.saw_catch_all = false
        self.covered_true = false
        self.covered_false = false
        self.variants = set()
        self.literals = []
        self.covered_lengths = set()
        self.covered_from = MAX_TRACKED_LENGTHS
    }
}

# Only an arm that binds unconditionally covers its shape; `[1, x]` still falls
# through for a list of two whose head is not 1.
fn _pattern_is_irrefutable(pat: Node?) -> bool {
    if pat is null { return false }
    return pat.kind == NodeKind.PatWildcard or pat.kind == NodeKind.PatBind
}

fn _sequence_arm_is_irrefutable(pat: Node) -> bool {
    let rest = pat.number("restIndex")
    for (position, element) in pat.child_list("elems").enumerate() {
        if position == rest { continue }
        if not _pattern_is_irrefutable(element) { return false }
    }
    return true
}

# `ConstValue.__eq__` would call two `None`s equal and the C's switch answers
# false, so the rule is written out rather than relied upon.
fn _same_const(a: ConstValue, b: ConstValue) -> bool {
    if a.kind != b.kind { return false }
    if a.kind == ConstKind.None { return false }
    if a.kind == ConstKind.Null { return true }
    return a.value == b.value
}

fn _pattern_covered(c: Ctx, pat: Node?, enum_decl: TypeDecl?, cov: _Coverage) -> bool {
    if pat is null { return false }
    match pat.kind {
        NodeKind.PatWildcard | NodeKind.PatBind => { return cov.saw_catch_all }

        NodeKind.PatLiteral => {
            let v = const_eval(pat.child("value"), null, c.consts)
            if v.kind == ConstKind.None { return false }
            if v.kind == ConstKind.Bool {
                return if v.as_bool() { cov.covered_true } else { cov.covered_false }
            }
            for seen in cov.literals {
                if _same_const(seen, v) { return true }
            }
            return false
        }

        NodeKind.PatEnum | NodeKind.PatClass => {
            let variant_name = pat.opt_text("variantName")
            if enum_decl is null or variant_name is null { return false }
            let vi = _field_index(enum_decl, variant_name)
            if vi < 0 or vi >= MAX_TRACKED_VARIANTS { return false }
            # Only a payload-free variant is fully covered by an earlier arm; a
            # payload arm may have narrowed on its sub-patterns.
            return vi in cov.variants and pat.child_list("subPatterns").len() == 0
        }

        NodeKind.PatOr => {
            let alternatives = pat.child_list("elems")
            for alternative in alternatives {
                if not _pattern_covered(c, alternative, enum_decl, cov) { return false }
            }
            return alternatives.len() > 0
        }

        _ => { return false }
    }
}

fn _cover_pattern(c: Ctx, pat: Node?, enum_decl: TypeDecl?, cov: _Coverage) -> void {
    if pat is null { return }
    match pat.kind {
        NodeKind.PatWildcard | NodeKind.PatBind => { cov.saw_catch_all = true }

        NodeKind.PatLiteral => {
            let v = const_eval(pat.child("value"), null, c.consts)
            if v.kind == ConstKind.None { return }
            if v.kind == ConstKind.Bool {
                if v.as_bool() {
                    cov.covered_true = true
                } else {
                    cov.covered_false = true
                }
                return
            }
            # Past the cap the C stops recording, so a later duplicate of an
            # untracked literal is never reported unreachable.
            if cov.literals.len() >= MAX_TRACKED_LITERALS { return }
            cov.literals.push(v)
        }

        NodeKind.PatEnum | NodeKind.PatClass => {
            let variant_name = pat.opt_text("variantName")
            if enum_decl is null or variant_name is null { return }
            let vi = _field_index(enum_decl, variant_name)
            if vi >= 0 and vi < MAX_TRACKED_VARIANTS { cov.variants.add(vi) }
        }

        NodeKind.PatList | NodeKind.PatTuple => {
            if not _sequence_arm_is_irrefutable(pat) { return }
            var fixed = pat.child_list("elems").len()
            if pat.number("restIndex") >= 0 {
                fixed -= 1  # `...rest` matches zero or more
                if fixed < cov.covered_from { cov.covered_from = fixed }
            } elif fixed < MAX_TRACKED_LENGTHS {
                cov.covered_lengths.add(fixed)
            }
        }

        NodeKind.PatOr => {
            for alternative in pat.child_list("elems") {
                _cover_pattern(c, alternative, enum_decl, cov)
            }
        }

        _ => { }
    }
}

fn _report_non_exhaustive(
    c: Ctx,
    node: Node,
    subject: Type?,
    enum_decl: TypeDecl?,
    cov: _Coverage
) -> void {
    if cov.saw_catch_all { return }

    if enum_decl is not null {
        var missing: list[str] = []
        let fields = _layout_fields(enum_decl)
        var at = 0
        while at < enum_decl.variant_count and at < MAX_TRACKED_VARIANTS and at < fields.len() {
            if at not in cov.variants {
                missing.push(f"`{enum_decl.name}.{fields[at].name}`")
            }
            at += 1
        }
        if missing.len() == 0 { return }
        let listed = ", ".join(missing)
        let d = c.error(
            E0501_NON_EXHAUSTIVE_MATCH,
            node.span,
            f"non-exhaustive match: {listed} not covered"
        )
        d.label(enum_decl.span, f"`{enum_decl.name}` is declared here")
        let plural = if missing.len() == 1 { "" } else { "s" }
        d.with_help(f"add the missing arm{plural}, or a `_ => ...` arm")
        return
    }

    if subject is not null and
        subject.kind == TyKind.Bool and
        (not cov.covered_true or not cov.covered_false) {
        let absent = if cov.covered_true { "false" } else { "true" }
        let d = c.error(
            E0501_NON_EXHAUSTIVE_MATCH,
            node.span,
            f"non-exhaustive match: `{absent}` is not covered"
        )
        d.with_help("add the missing arm, or a `_ => ...` arm")
    }
}

# An open `[..., ...rest]` arm takes everything from `covered_from` up, so only
# the shorter lengths still need an exact arm. Only a list or tuple subject can
# be exhausted this way: an `any` need not be a sequence at all.
fn _sequence_coverage_complete(subject: Type?, cov: _Coverage) -> bool {
    if subject is null { return false }
    if subject.kind != TyKind.List and subject.kind != TyKind.Tuple { return false }
    if cov.covered_from >= MAX_TRACKED_LENGTHS { return false }
    var length = 0
    while length < cov.covered_from {
        if length not in cov.covered_lengths { return false }
        length += 1
    }
    return true
}

#: The type of a `match`, and every diagnostic its arms and coverage produce.
#:
#: `jaiChkMatchExpr`, and the shape of the installed `Ctx.match_expr`. In
#: statement position the answer is `void`; in expression position the arms are
#: joined, and a match that may fall through yields `T?` on the missing arm.
#:
#: A statically unknown subject — `any`, or an enum this module only imported —
#: is not evidence of a missing arm, so the arms are taken at their word rather
#: than infecting every result with `| null`.
pub fn match_expr(c: Ctx, node: Node?, is_statement_position: bool) -> Type {
    let u = c.universe
    if node is null { return u.t_any }

    let subject = value(c, node.child("subject"))
    var enum_decl = _type_decl_of(subject)
    if enum_decl is not null and not enum_decl.is_enum { enum_decl = null }
    if enum_decl is not null { c.layout_decl(c, enum_decl) }

    let cov = _Coverage()
    var result: Type? = null

    for arm in node.records("arms") {
        let arm_pattern: Node? = arm.get("pattern", null)
        let arm_guard: Node? = arm.get("guard", null)
        let arm_body: Node? = arm.get("body", null)
        let arm_span: Span = arm.get("span", Span.none())

        if _pattern_covered(c, arm_pattern, enum_decl, cov) {
            let d = c.error(
                E0500_UNREACHABLE_ARM,
                arm_span,
                "unreachable match arm: an earlier arm already matches"
            )
            d.with_help("remove it, or move it above the arm that subsumes it")
        }

        let mark = c.narrow_mark()
        pattern(c, arm_pattern, subject)
        if arm_guard is not null { condition(c, arm_guard, "a match guard") }

        let body = if arm_body is null { u.t_void } else { expr(c, arm_body) }
        c.narrow_restore(mark)

        if not is_statement_position and not _is_void(c, body) and not _is_never(body) {
            result = if result is null { body } else { join(u, result, body) }
        }

        # A guarded arm does not cover its pattern: the guard may be false.
        if arm_guard is null { _cover_pattern(c, arm_pattern, enum_decl, cov) }
    }

    _report_non_exhaustive(c, node, subject, enum_decl, cov)
    if is_statement_position { return u.t_void }
    if result is null { return u.t_any }
    let total = cov.saw_catch_all or
        enum_decl is not null or
        _is_any(subject) or
        _sequence_coverage_complete(subject, cov)
    return if total { result } else { optional_of(u, result) }
}

# `declSymbol`: the C reads `AST_VAR_DECL.symbol` and falls back to the
# pattern's. `resolve.jai` binds the symbol to both nodes for a plain binding.
fn _decl_symbol(c: Ctx, node: Node) -> Symbol? {
    let own = c.resolution.symbol_of(node)
    if own is not null { return own }
    let pat = node.child("pattern")
    if pat is not null and pat.kind == NodeKind.PatBind { return c.resolution.symbol_of(pat) }
    return null
}

# An unannotated `var` initialised to null would otherwise be pinned to the
# null type and reject every later assignment; the binding is dynamic.
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

    # Passing the bag is the C's `gJaiCheck.constRequired`: here and only here
    # is a failed fold worth E0900 or E0901.
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

    # Keep the folded value where every use can inline it. A Jaithon `Symbol`
    # has no `isConstFolded`/`constValue`, so `Ctx.consts` is the stand-in,
    # keyed by name rather than identity. Both rewrites are dumped, as in the C:
    # the synthetic literal produces a `fold` record of its own at the
    # initialiser's span before the initialiser produces the second.
    let literal = Node.null_lit().with_span(init.span)
    c.set_type(literal, const_replace(c.universe, literal, v))
    c.set_type(init, const_replace(c.universe, init, v))
    if symbol is not null { c.consts[symbol.name] = literal }
}

fn _check_var_decl(c: Ctx, node: Node) -> void {
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

    let settled = _or_any(c, bound)
    c.set_type(node, settled)

    let symbol = _decl_symbol(c, node)
    c.set_symbol_type(symbol, settled)
    let pat = node.child("pattern")
    if pat is not null { pattern(c, pat, settled) }

    let kind: DeclKind = node.get("varDeclKind")
    if kind == DeclKind.Const { _fold_const_decl(c, node, symbol) }
}

# The literal conversion has to happen before the value is typed, or an int
# literal in a float field would be reported and then silently converted.
fn _check_stored_value(c: Ctx, source: Node?, want: Type?) -> void {
    if want is not null and not _is_any(want) { c.apply_context(c, source, want) }
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
    if found is null { found = _type_decl_of(value(c, object)) }
    if found is null {
        # An `any` receiver, a module or a builtin: checked at run time.
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
        # A `let` field is settable from inside its own class — that is what
        # `init` is for — and nowhere else.
        if field.is_let and not static_access and c.current_class != decl {
            let d = c.error(
                E0301_ASSIGN_TO_IMMUTABLE,
                target.span,
                f"`{name}` is declared `let` and cannot be reassigned"
            )
            d.with_help("declare it `var` if it needs to change after `init`")
        }

        c.set_type(target, _or_any(c, field.ty))
        _check_stored_value(c, source, c.type_of(target))
        return
    }

    let setter = find_accessor(decl, name, true)
    if setter is not null {
        c.set_type(target, _or_any(c, setter.ty))
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
    # A compound assignment reads the target first, so it is typed as an
    # operand of the operator before it is typed as a destination.
    let current = expr(c, target)
    var rhs = value(c, source)
    if _is_any(current) or _is_any(rhs) or _is_never(current) or _is_never(rhs) { return }

    let op = node.op()
    # `x += n` widens an int operand to a float target, as `x = x + n` would.
    if widens_operands(op) and current.kind == TyKind.Float and rhs.kind == TyKind.Int {
        c.widen_to_float(c, source, u.t_float)
        rhs = u.t_float
    }

    # The other direction cannot: the result writes back into the target, and a
    # float does not fit an int.
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
    let shape = if _is_any(container) { null } else { is_indexable(u, container) }
    if shape is not null {
        let (want_index, want_element) = shape
        if not _is_any(index) {
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
    } elif not _is_any(container) and _type_decl_of(container) is null {
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
    c.set_type(target, _or_any(c, element))
}

fn _check_assign(c: Ctx, node: Node) -> void {
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
            # The destination is the declaration, not the flow-narrowed view: in
            # `if v != null { ... }` a `v = maybeNull()` is still a legal store.
            if symbol is not null and c.symbol_type(symbol) is not null {
                want = _or_any(c, c.declared_type(symbol))
            }
            c.set_type(target, _or_any(c, want))
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
            # Destructuring assignment: `(a, b) = point`.
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
        # Declarations are hoisted by the resolver, so only real statements
        # after a diverging one are dead.
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
    # The false edge is collected separately rather than by inverting the true
    # edge: `a or b` yields nothing when true, but `not (a or b)` yields both
    # facts, which is the `if x is null or x <= 0 { return }` guard.
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

    # `if x == null { return }` narrows everything after it: the facts stay
    # applied and the enclosing block restores them when it ends.
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

    # The C types `AstNode.forLoop.iterSymbol` here. The Jaithon `For` schema
    # has no such field and `resolve.jai` creates no such symbol, so there is
    # nothing to type and this is where that step would go.
    let body = node.child("body")
    if body is not null { stmt(c, body) }
    c.narrow_restore(mark)
}

fn _check_return(c: Ctx, node: Node) -> void {
    let want = c.current_return_type
    let returned = node.child("value")

    if returned is null {
        if want is not null and
            not _is_any(want) and
            not _is_void(c, want) and
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

    if want is not null and _is_void(c, want) {
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
    if _is_any(t) or _is_never(t) { return }
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
            caught = _or_any(c, c.lookup_type_name(c, "Error"))
        }
        c.set_symbol_type(c.resolution.symbol_of(handler), _or_any(c, caught))
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
    if _is_any(t) or t.kind == TyKind.Str { return }
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

        # The only place a `void` value is allowed: nothing consumes it.
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

        # Loop nesting and labels are the resolver's business (E0203, E0204);
        # what these two contribute here is divergence, which
        # `ast.always_diverges` answers.
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
        return _or_any(c, c.resolve_ast_type(c, written))
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
        params.push(_or_any(c, declared))
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
    let settled = _or_any(c, declared)
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
            bound = _decl_type(c, owner)
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
            if not _is_any(returns) { c.apply_context(c, body, returns) }
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
        not _is_void(c, returns) and
        not _is_any(returns) and
        not _is_never(returns) and
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

# A lambda is never a method and has no owner: the two-argument case `Ctx` wants.
fn _check_lambda(c: Ctx, fn_node: Node?) -> void {
    function(c, fn_node, null, false)
}

fn _lambda_type(c: Ctx, fn_node: Node?) -> Type {
    return function_type(c, fn_node, false)
}
