#: A `match`: its arms, the coverage they build up between them, and the type
#: the whole form answers.
#:
#: Coverage is a running summary rather than a decision procedure. `_Coverage`
#: remembers what earlier arms took — a catch-all, the two `bool` values, enum
#: variant ordinals, literal values, sequence lengths — so a later arm can be
#: called unreachable (`E0500`) and a missing one can be named (`E0501`). Every
#: tally is capped, because the C's are stack arrays and the cap is observable:
#: past it the C stops reporting and so must this.
#:
#: A guarded arm covers nothing. It may not run, so `_cover_pattern` is skipped
#: for it while `_pattern_covered` still runs against what came before.
#:
#: The result type. In statement position it is `void`. Otherwise the arm bodies
#: are joined, and the join is wrapped in `T?` unless the match is total —
#: a catch-all, an enum whose variants are all taken, a subject this module
#: cannot see into, or a sequence whose lengths are all accounted for. An
#: unknown subject counts as total on purpose: `any` is not evidence of a
#: missing arm, and treating it as one would infect every result with `| null`.
#:
#: The file is named `matching` and not `match` because `match` is a keyword,
#: and a keyword cannot be a segment of a module path.
from jaithon.ast import Node, NodeKind, Span
from jaithon.compile.check.ctx import Ctx, is_any, is_never, is_void, type_decl_of
from jaithon.compile.check.expr import condition, expr, value
from jaithon.compile.check.fold import ConstKind, ConstValue, const_eval
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.relate import join
from jaithon.compile.check.stmt.codes import E0500_UNREACHABLE_ARM, E0501_NON_EXHAUSTIVE_MATCH
from jaithon.compile.check.stmt.codes import MAX_TRACKED_LENGTHS, MAX_TRACKED_LITERALS
from jaithon.compile.check.stmt.codes import MAX_TRACKED_VARIANTS
from jaithon.compile.check.stmt.fields import _field_index, _layout_fields
from jaithon.compile.check.stmt.pattern import pattern
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.union import optional_of

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
                fixed -= 1
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
    var enum_decl = type_decl_of(subject)
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

        if not is_statement_position and not is_void(c, body) and not is_never(body) {
            result = if result is null { body } else { join(u, result, body) }
        }

        if arm_guard is null { _cover_pattern(c, arm_pattern, enum_decl, cov) }
    }

    _report_non_exhaustive(c, node, subject, enum_decl, cov)
    if is_statement_position { return u.t_void }
    if result is null { return u.t_any }
    let total = cov.saw_catch_all or
        enum_decl is not null or
        is_any(subject) or
        _sequence_coverage_complete(subject, cov)
    return if total { result } else { optional_of(u, result) }
}
