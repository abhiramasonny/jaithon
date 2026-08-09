#: The five type-shape predicates: is this a number, can it be null, can it be
#: called, looped over, indexed?
#:
#: A port of the `Predicates` section of `src/sema/types.c`. Each one is the
#: question a syntactic form has to ask before it means anything: `f(...)` asks
#: `is_callable`, `for x in e` asks `is_iterable`, `e[i]` asks `is_indexable`,
#: and `x?.y`, `x ?? y` and a function that can fall off its end all ask
#: `is_optional`.
#:
#: **The last two answer with a value where the C answers with a flag.**
#: `jaiTypeIsIterable(t, &elem)` returns a bool and writes the element type
#: through a pointer, so a caller that reads the pointer without the bool reads
#: whatever was there before. Here `is_iterable` answers the element type, or
#: null when `t` cannot be looped over, and `is_indexable` answers
#: `(index, result)` or null. Nothing is lost in the trade: the C never accepts
#: a type and then writes a null through either pointer.
#:
#: The universe is a parameter to those two alone, because they are the only
#: ones that *build* an answer rather than inspect one — `str` iterates as
#: `str`, `bytes` as `int` — and an answer has to come from the same intern
#: table as everything else so that `is_iterable(u, u.t_bytes) == u.t_int` stays
#: a comparison rather than a walk.
#:
#: So is `join`, for a duller reason: a tuple answers the join of its members,
#: and the C calls `jaiTypeJoin` for that. The default is `relate.join`, which
#: is that function — so `(Circle, Shape)` iterates as `Shape`, the way the C's
#: checker binds it, rather than as `Circle | Shape`. It stays a parameter
#: rather than a plain call because the element type is the one thing these two
#: take from elsewhere instead of deciding, and naming it in the signature is
#: what keeps that visible; `operator.jai` carries the same seam for the same
#: reason.
#:
#: The member lookups are `nominal.find_member` rather than a walk of their own.
#: That is the port of `declFindMember`, cap and order and all, and the three
#: dunder questions below want exactly what it answers: fields are not searched,
#: because `iter` and `__getitem__` are methods and a field holding a function
#: is not one. Only the C's out-parameter is unused here — no predicate reads
#: the member's type, only whether one exists.
#:
#: `resolveAlias` is not ported. The C opens all five with it; `TyKind.Alias`
#: exists for ordinal fidelity and nothing in either front end constructs one
#: (see `kinds.jai`), so the loop would never take a hop. Every other file in
#: this directory leaves it out for the same reason.
#:
#: These are shape questions and not the checker's final word. `check_expr.c`
#: lays a declaration out before asking — an imported class that has not been
#: laid out yet has no members to find — and it has fallbacks of its own for a
#: class whose declaration this compilation unit cannot see. That layer is
#: Phase 2's; this file answers from the type alone.
#:
#: ```
#: let u = Universe()
#: is_numeric(u.t_float)      # true
#: is_optional(u.t_any)       # true
#: is_callable(u.t_str)       # false
#: ```
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.nominal import find_member
from jaithon.compile.check.relate import join as join_types
from jaithon.compile.check.ty import Type
from jaithon.compile.check.universe import Universe

#: Is `t` `int` or `float`?
#:
#: `bool` is not numeric, even though the VM stores it as an integer: the
#: language has no truthiness and no bool arithmetic, so the two kinds this
#: admits are the two the arithmetic rows of `jaiTypeBinaryResult` admit. `any`
#: is not numeric either — anything mixed with `any` is already dynamic by the
#: time an operator asks.
#:
#: Nothing calls it. It is declared in `src/sema/types.h` and defined in
#: `types.c`, and no caller in either front end uses it today: the checker
#: reaches the same conclusion through the operator table, which tests the
#: kinds directly. It is ported because the surface is what is being ported,
#: and a predicate this small is cheaper to carry than to re-derive when the
#: first caller appears.
#:
#: ```
#: let u = Universe()
#: is_numeric(u.t_int)      # true
#: is_numeric(u.t_float)    # true
#: is_numeric(u.t_bool)     # false
#: is_numeric(null)         # false
#: ```
pub fn is_numeric(t: Type?) -> bool {
    if t is null { return false }
    return t.kind == TyKind.Int or t.kind == TyKind.Float
}

#: Can a value of `t` be null?
#:
#: `null` and `any` both answer true with no union around them, which is the
#: whole point of the two lines the C spends on them: `null` is nothing but
#: null, and `any` admits it along with everything else. Without those two
#: cases `x ?? y` on a dynamic binding would be reported as a redundant
#: coalesce.
#:
#: A union answers true when any member is `Null`, and one level of members is
#: enough — `union_of` flattens nested unions out before interning, so there is
#: no union hiding inside a union for this loop to miss.
#:
#: ```
#: from jaithon.compile.check.union import optional_of, union_of
#: let u = Universe()
#: is_optional(optional_of(u, u.t_int))          # true
#: is_optional(u.t_null)                         # true
#: is_optional(u.t_any)                          # true
#: is_optional(union_of(u, [u.t_int, u.t_str]))  # false
#: is_optional(u.t_int)                          # false
#: ```
pub fn is_optional(t: Type?) -> bool {
    if t is null { return false }
    if t.kind == TyKind.Null or t.kind == TyKind.Any { return true }
    if t.kind != TyKind.Union { return false }
    for member in t.args {
        if member.kind == TyKind.Null { return true }
    }
    return false
}

#: Can `t` appear before a `(`?
#:
#: A class is callable because the class object *is* its constructor (spec
#: §7.1), and an enum because that is how a variant is built — neither needs a
#: dunder. A trait needs one: it has no constructor of its own, so `__call__` is
#: the only thing that could give the call a meaning.
#:
#: ```
#: from jaithon.compile.check.universe import Universe, fn_of, named
#: let u = Universe()
#: is_callable(fn_of(u, [u.t_int], u.t_str))     # true
#: is_callable(named(u, TyKind.Class, "Box"))    # true
#: is_callable(named(u, TyKind.Trait, "Shape"))  # false
#: is_callable(u.t_any)                          # true
#: is_callable(u.t_int)                          # false
#: ```
pub fn is_callable(t: Type?) -> bool {
    if t is null { return false }
    match t.kind {
        TyKind.Fn | TyKind.Any | TyKind.Class | TyKind.Enum => { return true }
        TyKind.Trait => { return find_member(t.decl, "__call__") is not null }
        _ => { return false }
    }
}

#: What `for x in e` binds, or null when `e` cannot be looped over.
#:
#: A dict iterates its **keys**, not its pairs, which is why `Dict` answers
#: `args[0]` and not a two-member tuple. A `str` iterates as `str`, one scalar
#: at a time rather than one byte; `bytes` is the one that yields `int`.
#:
#: A class, trait or enum is iterable when it declares `iter` or `__iter__`,
#: and the element is `any`. The precise element does exist — it is the `T` of
#: the `Iterator[T]` that `iter` returns — but reading it means opening the
#: trait instantiation, which nothing in this directory can do, so the C widens
#: to `any` rather than guessing and this follows it.
#:
#: **A union is not iterable, even when every member is.** The C's switch has
#: no union case, so `list[int] | list[str]` falls to the default and answers
#: no. A loop over one has to be narrowed first.
#:
#: ```
#: from jaithon.compile.check.universe import Universe, dict_of, list_of, set_of
#: let u = Universe()
#: is_iterable(u, list_of(u, u.t_int)) == u.t_int             # true
#: is_iterable(u, set_of(u, u.t_str)) == u.t_str              # true
#: is_iterable(u, dict_of(u, u.t_str, u.t_int)) == u.t_str    # true
#: is_iterable(u, u.t_bytes) == u.t_int                       # true
#: is_iterable(u, u.t_any) == u.t_any                         # true
#: is_iterable(u, u.t_int) is null                            # true
#: ```
pub fn is_iterable(
    u: Universe,
    t: Type?,
    join: fn(Universe, Type?, Type?) -> Type = join_types
) -> Type? {
    if t is null { return null }
    match t.kind {
        TyKind.List | TyKind.Set => { return t.args[0] }
        # Iterating a dict yields its keys.
        TyKind.Dict => { return t.args[0] }
        TyKind.Tuple => { return _join_all(u, t.args, join) }
        # One scalar at a time, not one byte.
        TyKind.Str => { return u.t_str }
        TyKind.Bytes | TyKind.Range => { return u.t_int }
        TyKind.Any => { return u.t_any }
        TyKind.Class | TyKind.Trait | TyKind.Enum => {
            # Two lookups rather than one `or`, so the second walk is skipped
            # when the first answers — which is what the C's `&&` does.
            if find_member(t.decl, "iter") is not null { return u.t_any }
            if find_member(t.decl, "__iter__") is not null { return u.t_any }
            return null
        }
        _ => { return null }
    }
}

#: What `e[i]` wants for `i` and hands back, or null when `e` cannot be indexed.
#:
#: The index is `int` for every container but a dict, which wants its own key
#: type, and `any`, where both halves are `any`. A user type with `__getitem__`
#: is also `(any, any)`: the dunder's signature is not consulted, so the
#: subscript is unchecked and the result is dynamic.
#:
#: **A set is not indexable.** It is the one container that is iterable and not
#: subscriptable, and it reaches the same default a `bool` does. A union is not
#: indexable either, for the same reason it is not iterable.
#:
#: A tuple answers the join of its members, because nothing here can see the
#: subscript. That is the weakest of the answers and it is deliberate: a caller
#: that *can* see a constant index does better on its own, and `checkIndex` in
#: `src/sema/check_expr.c` resolves `pair[0]` on a `(str, int)` to `str` before
#: it ever gets here.
#:
#: ```
#: from jaithon.compile.check.universe import Universe, dict_of, list_of, set_of
#: let u = Universe()
#: is_indexable(u, list_of(u, u.t_str)) == (u.t_int, u.t_str)              # true
#: is_indexable(u, dict_of(u, u.t_str, u.t_int)) == (u.t_str, u.t_int)     # true
#: is_indexable(u, u.t_bytes) == (u.t_int, u.t_int)                        # true
#: is_indexable(u, set_of(u, u.t_int)) is null                             # true
#: ```
pub fn is_indexable(
    u: Universe,
    t: Type?,
    join: fn(Universe, Type?, Type?) -> Type = join_types
) -> (Type, Type)? {
    if t is null { return null }
    match t.kind {
        TyKind.List => { return (u.t_int, t.args[0]) }
        TyKind.Dict => { return (t.args[0], t.args[1]) }
        TyKind.Tuple => { return (u.t_int, _join_all(u, t.args, join)) }
        TyKind.Str => { return (u.t_int, u.t_str) }
        TyKind.Bytes | TyKind.Range => { return (u.t_int, u.t_int) }
        TyKind.Any => { return (u.t_any, u.t_any) }
        TyKind.Class | TyKind.Trait | TyKind.Enum => {
            if find_member(t.decl, "__getitem__") is not null { return (u.t_any, u.t_any) }
            return null
        }
        _ => { return null }
    }
}

# `joinAll` in the C: the join folded over the members starting from `never`.
# Only a tuple needs it, and only because neither predicate above can see which
# member a subscript or a loop step selects.
#
# Folded rather than collapsed into one `union_of` so that the cap inside the
# join fires where the C's fires, per step and not once at the end.
#
# The seed is what makes the join's early arms load-bearing here rather than a
# shortcut. A join that hands an operand straight back is not building a least
# upper bound and must not be able to widen one: every fold starts at `never`,
# so a join without a `Never` arm — or without the subtype test that stands in
# for one — would take a tuple whose single member is an already-interned
# nine-member union through `union_of`, trip the cap on the union it was merely
# carrying, and answer `any` on the very first step, where the C answers with
# the union itself.
fn _join_all(u: Universe, members: list[Type], join: fn(Universe, Type?, Type?) -> Type) -> Type {
    var result: Type = u.t_never
    for member in members { result = join(u, result, member) }
    return result
}
