#: Binding a pattern against the type it destructures, and reporting what does
#: not fit: arity, an unknown type or variant, a field the class does not have
#: or may not show, a literal that can never equal the subject, and one name
#: bound twice.
#:
#: The walk is total. Every arm ends by giving the pattern node a type, because
#: `emit` reads the type of a `PatBind` to size its slot, and a pattern that
#: failed its check still has to be typed as something — the subject, widened to
#: `any` where nothing better is known.
#:
#: An or-pattern gets a **fresh** `_BindSet` per alternative on purpose: `A(x) |
#: B(x)` binds `x` once from the reader's point of view, so the duplicate check
#: must not see the two arms as one pattern.
#:
#: What is NOT here: whether the arms of a `match` cover the subject. A pattern
#: knows only itself; `matching.jai` accumulates coverage across arms and owns
#: `E0500` and `E0501`.
from jaithon.ast import Node, NodeKind, Span, is_expression
from jaithon.compile.check.assign import assignable
from jaithon.compile.check.ctx import Ctx, is_any, or_any, type_decl_of
from jaithon.compile.check.expr import E0400_TYPE_MISMATCH, E0410_UNKNOWN_MEMBER
from jaithon.compile.check.expr import E0701_PRIVATE_ACCESS, expr, value
from jaithon.compile.check.expr import visibility_word, visible_from
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.nominal import owner_of_field
from jaithon.compile.check.render import render
from jaithon.compile.check.stmt.codes import E0402_UNKNOWN_TYPE, E0502_PATTERN_ARITY
from jaithon.compile.check.stmt.codes import E0503_DUPLICATE_BINDING_IN_PATTERN, MAX_BINDS
from jaithon.compile.check.stmt.fields import _field_index, _layout_fields
from jaithon.compile.check.suggest import suggest_member
from jaithon.compile.check.ty import Type
from jaithon.compile.check.universe import list_of

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
        let from_subject = type_decl_of(subject)
        if from_subject is not null and from_subject.name == type_name { decl = from_subject }
    }
    if decl is null {
        if not is_any(subject) {
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
        if not is_any(subject) {
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
            field_type = or_any(c, field.ty)
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
    let subject = or_any(c, raw_subject)

    match pat.kind {
        NodeKind.PatWildcard => { }

        NodeKind.PatBind => {
            let written = pat.type_field("type")
            var bound: Type? = subject
            if written is not null { bound = c.resolve_ast_type(c, written) }
            _record_binding(c, binds, pat.opt_text("name"), pat.span)
            c.set_symbol_type(c.resolution.symbol_of(pat), or_any(c, bound))
            c.set_type(pat, or_any(c, bound))
            return
        }

        NodeKind.PatLiteral => {
            let literal_node = pat.child("value")
            if literal_node is not null {
                c.apply_context(c, literal_node, subject)
                let lit = value(c, literal_node)
                let (forwards, _fg) = assignable(u, lit, subject)
                let (backwards, _bg) = assignable(u, subject, lit)
                if not is_any(subject) and not is_any(lit) and not forwards and not backwards {
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
