#: Union normalisation: what makes `A | B` and `B | A` the same type.
#:
#: A union is interned like anything else, so two unions are the same type only
#: when their member lists are identical. That makes normalisation
#: identity-affecting rather than cosmetic: the order this file puts members in
#: is the order they are stored in, and the order they are stored in is the
#: order `render` prints them in, which is the string `make sema-diff` compares
#: against the C. Every step below is therefore a port and not a choice.
#:
#: The order of operations is the C's, in `jaiTypeUnion`:
#:
#: 1. **Flatten** — a nested union contributes its members, not itself.
#: 2. **Absorb** — one `any` member and the whole union is `any`.
#: 3. **Drop `never`** — the bottom type cannot widen a union it cannot inhabit.
#: 4. **Sort** by `compare_types` below.
#: 5. **Deduplicate** — adjacent equals collapse, which the sort makes complete.
#: 6. **Collapse** — no members is `never`, one member is that member with no
#:    union interned at all.
#:
#: ```
#: let u = Universe()
#: render(union_of(u, [u.t_str, u.t_int]))              # "int | str"
#: render(union_of(u, [u.t_any, u.t_int]))              # "any"
#: render(union_of(u, [u.t_int, u.t_int]))              # "int"
#: render(union_of(u, [u.t_null, u.t_int]))             # "int?"
#: ```
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.ty import Type
from jaithon.compile.check.universe import Universe

#: Order two types the way the C's `typeCompare` does.
#:
#: Kind first, then name, then arity, then arguments element-wise, then the
#: return type, then the function flags. Negative when `a` sorts first, zero
#: when they are the same type.
#:
#: Named types are compared by kind and name before their arguments are
#: reached, which is what stops a recursive class — `class Node { next: Node }`
#: — from recursing forever here.
pub fn compare_types(a: Type?, b: Type?) -> int {
    if a == null and b == null { return 0 }
    if a == null { return -1 }
    if b == null { return 1 }
    if a.ord != b.ord { return if a.ord < b.ord { -1 } else { 1 } }

    let a_name = a.name
    let b_name = b.name
    if a_name != b_name {
        if a_name == null { return -1 }
        if b_name == null { return 1 }
        if a_name < b_name { return -1 }
        if b_name < a_name { return 1 }
    }

    if a.arg_count() != b.arg_count() {
        return if a.arg_count() < b.arg_count() { -1 } else { 1 }
    }
    for (position, argument) in a.args.enumerate() {
        let order = compare_types(argument, b.args[position])
        if order != 0 { return order }
    }

    let order = compare_types(a.ret, b.ret)
    if order != 0 { return order }

    if a.fn_flags != b.fn_flags {
        return if a.fn_flags < b.fn_flags { -1 } else { 1 }
    }
    return 0
}

# A nested union contributes its members rather than itself, so that
# `(int | str) | float` and `int | str | float` intern as one type.
fn _collect(out: list[Type], t: Type?) -> void {
    if t == null { return }
    if t.kind == TyKind.Union {
        for member in t.args { _collect(out, member) }
        return
    }
    out.push(t)
}

#: The union of `members`, normalised.
#:
#: Answers `never` for an empty list, because a union of nothing is a type no
#: value inhabits.
pub fn union_of(u: Universe, members: list[Type?]) -> Type {
    if members.len() == 0 { return u.t_never }

    var flat: list[Type] = []
    for member in members { _collect(flat, member) }

    # `any` absorbs: a union that can hold anything is `any`. `never` is
    # dropped for the mirror reason.
    var live: list[Type] = []
    for member in flat {
        if member.kind == TyKind.Any { return u.t_any }
        if member.kind == TyKind.Never { continue }
        live.push(member)
    }

    # Insertion sort, as the C does: union members are few, and this is
    # deterministic in a way a general sort's tie-breaking need not be.
    var sorted: list[Type] = []
    for member in live {
        var at = sorted.len()
        while at > 0 and compare_types(sorted[at - 1], member) > 0 {
            at -= 1
        }
        sorted.insert(at, member)
    }

    # Equal members are adjacent after the sort, so one comparison per member
    # is a complete deduplication.
    var unique: list[Type] = []
    for member in sorted {
        if unique.len() > 0 and compare_types(unique[unique.len() - 1], member) == 0 {
            continue
        }
        unique.push(member)
    }

    if unique.len() == 0 { return u.t_never }
    if unique.len() == 1 { return unique[0] }
    return u.intern(TyKind.Union, null, unique)
}

#: `t | null`, which is what a written `T?` means.
pub fn optional_of(u: Universe, t: Type?) -> Type {
    if t == null { return u.t_any }
    return union_of(u, [t, u.t_null])
}
