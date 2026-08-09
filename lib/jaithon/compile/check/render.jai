#: The canonical text of a type.
#:
#: `make sema-diff` compares this output against the C front end, so a single
#: character of disagreement is a false divergence on every program that
#: mentions the type. Ported line for line from `renderType`, `renderFn` and
#: `renderUnion` in `src/sema/types.c`.
from jaithon.compile.check.kinds import MAX_RENDER_DEPTH, PRIMITIVE_SPELLING, TyKind
from jaithon.compile.check.ty import Type

#: The canonical text of `t`. Null renders `<unknown>` rather than raising:
#: diagnostics still have to print a type that failed to be computed.
pub fn render(t: Type?) -> str {
    var out: list[str] = []
    _render(out, t, 0)
    return "".join(out)
}

fn _render(out: list[str], t: Type?, depth: int) -> void {
    if t == null {
        out.push("<unknown>")
        return
    }
    # Depth starts at 0 and the guard is strictly greater, so the first `...`
    # appears at the thirty-fourth level. The off-by-two is the C's and is kept.
    if depth > MAX_RENDER_DEPTH {
        out.push("...")
        return
    }

    if t.kind in PRIMITIVE_SPELLING {
        out.push(PRIMITIVE_SPELLING[t.kind])
        return
    }

    match t.kind {
        TyKind.List => {
            out.push("list[")
            _render(out, t.args[0], depth + 1)
            out.push("]")
        }
        TyKind.Set => {
            out.push("set[")
            _render(out, t.args[0], depth + 1)
            out.push("]")
        }
        TyKind.Dict => {
            out.push("dict[")
            _render(out, t.args[0], depth + 1)
            out.push(", ")
            _render(out, t.args[1], depth + 1)
            out.push("]")
        }
        TyKind.Tuple => {
            # A one-member tuple is `(int)`, no trailing comma; an empty one `()`.
            out.push("(")
            for (position, member) in t.args.enumerate() {
                if position > 0 { out.push(", ") }
                _render(out, member, depth + 1)
            }
            out.push(")")
        }
        TyKind.Fn => { _render_fn(out, t, depth) }
        TyKind.Union => { _render_union(out, t, depth) }
        TyKind.Module => {
            out.push("module ")
            out.push(t.name ?? "<anonymous>")
        }
        _ => { _render_named(out, t, depth) }
    }
}

# The `_` arm: Class, Trait, Enum, GenericParam and Alias all render alike.
fn _render_named(out: list[str], t: Type, depth: int) -> void {
    out.push(t.name ?? "<anonymous>")
    if t.arg_count() > 0 {
        out.push("[")
        for (position, argument) in t.args.enumerate() {
            if position > 0 { out.push(", ") }
            _render(out, argument, depth + 1)
        }
        out.push("]")
    }
}

# A function consumes no nesting level of its own, only its parts do, so this
# takes the caller's depth rather than depth + 1.
fn _render_fn(out: list[str], t: Type, depth: int) -> void {
    let count = t.arg_count()
    let kw_index = if t.has_kwrest() { count - 1 } else { -1 }
    var var_index = -1
    if t.is_variadic() {
        # With a kwrest parameter present, `...` sits one slot before the `**`.
        var_index = count - 1 - (if kw_index >= 0 { 1 } else { 0 })
    }

    out.push("fn(")
    for (position, parameter) in t.args.enumerate() {
        if position > 0 { out.push(", ") }
        # kwrest is tested first, and neither marker carries a trailing space.
        if position == kw_index { out.push("**") } elif position == var_index { out.push("...") }
        _render(out, parameter, depth + 1)
    }
    out.push(") -> ")

    let returns = t.ret
    if returns == null {
        out.push("any")
    } elif returns.kind == TyKind.Null {
        # `void` is not a distinct type: it is this spelling of null (spec §6).
        out.push("void")
    } else {
        _render(out, returns, depth + 1)
    }
}

# Also handed the caller's depth, for the same reason as a function.
fn _render_union(out: list[str], t: Type, depth: int) -> void {
    # `T | null` renders as the `T?` sugar, except when `T` is a function or a
    # union: `fn() -> int?` would read back as a function returning `int?`.
    if t.arg_count() == 2 {
        var other: Type? = null
        if t.args[0].kind == TyKind.Null {
            other = t.args[1]
        } elif t.args[1].kind == TyKind.Null {
            other = t.args[0]
        }
        if other != null and other.kind != TyKind.Fn and other.kind != TyKind.Union {
            _render(out, other, depth + 1)
            out.push("?")
            return
        }
    }

    # Intern order sorts null near the front; a trailing null parses the same.
    var first = true
    var has_null = false
    for member in t.args {
        if member.kind == TyKind.Null {
            has_null = true
            continue
        }
        if not first { out.push(" | ") }
        _render(out, member, depth + 1)
        first = false
    }
    if has_null {
        if not first { out.push(" | ") }
        out.push("null")
    }
}
