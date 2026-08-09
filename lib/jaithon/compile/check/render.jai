#: The canonical text of a type.
#:
#: This is the most constrained file in the port. Its output is the value the
#: sema dump carries, and `make sema-diff` compares those strings between the C
#: front end and this one, so a single character of disagreement is a false
#: divergence on every program that mentions the type. It is a port of
#: `renderType`, `renderFn` and `renderUnion` in `src/sema/types.c`, line for
#: line rather than in spirit.
#:
#: It depends only on `kinds` and `type` — not on the universe, not on any
#: relation — so it is a pure function of its argument and can be tested before
#: any semantics exist. That is deliberate: it is the highest-risk file, so it
#: is the one that gets verified first.
#:
#: The C returns a pointer into an eight-slot rotating buffer so that several
#: `%s` arguments in one `printf` stay live at once. That is a C lifetime trick
#: with no counterpart here and is not reproduced; this returns an owned `str`.
#:
#: ```
#: render(null)     # "<unknown>"
#: ```
from jaithon.compile.check.kinds import MAX_RENDER_DEPTH, PRIMITIVE_SPELLING, TyKind
from jaithon.compile.check.ty import Type

#: The canonical text of `t`.
#:
#: Null renders `<unknown>` rather than raising: rendering is what diagnostics
#: and dumps call, and a type that failed to be computed still has to print.
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
    # The top level is depth 0 and the guard is strictly greater, so the first
    # `...` appears at the thirty-fourth level rather than the thirty-second.
    # The off-by-two is the C's and is kept.
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
            # Parenthesised, not `tuple[...]`: a one-member tuple is `(int)`
            # with no trailing comma and an empty one is `()`.
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
            # Modules have no type syntax; naming them makes a mismatch read
            # clearly instead of as a bare kind.
            out.push("module ")
            out.push(t.name ?? "<anonymous>")
        }
        _ => { _render_named(out, t, depth) }
    }
}

# Class, Trait, Enum, GenericParam and Alias all render the same way: the name,
# then the arguments in brackets only when there are any.
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

# A function consumes no nesting level of its own; only its parts do, which is
# why this is handed the caller's depth rather than depth + 1.
fn _render_fn(out: list[str], t: Type, depth: int) -> void {
    let count = t.arg_count()
    let kw_index = if t.has_kwrest() { count - 1 } else { -1 }
    var var_index = -1
    if t.is_variadic() {
        # Keyed on whether a kwrest parameter is present, not on the flag, so a
        # function with both puts `...` one before the `**`.
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
        # No return type was written at all.
        out.push("any")
    } elif returns.kind == TyKind.Null {
        # A function returning null is what `-> void` declares (spec §6), and
        # `void` is not a distinct type: it is this spelling of null.
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

    # Members are in intern order, which sorts null near the front; a trailing
    # null reads better and parses the same.
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
