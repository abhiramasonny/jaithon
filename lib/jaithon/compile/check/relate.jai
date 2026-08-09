#: Equality, the join and narrowing — the relations that answer with a type.
#:
#: A port of `jaiTypeEquals`, `jaiTypeJoin` and `jaiTypeNarrow` in
#: `src/sema/types.c`, together with the `joinAll` fold that sits beside them.
#:
#: **`jaiTypeIsSubtype` is the fourth of that group and is not here.** It is one
#: bit read off `assignable` — accepted, and unguarded — so it lives in
#: `assign.jai` next to the function it reads, and this file imports it from
#: there. Spelling `accepted and not guard` in two places would be two things to
#: keep in step every time the guard rule moves, and the guard rule is the part
#: of the type system that decides which instructions get emitted.
#:
#: **That one bit is why everything below is asymmetric.** `assignable` answers
#: both whether a value fits and whether it needs a run-time `TYPE_GUARD` to;
#: subtyping is the half that needs no guard. So `int` is a subtype of `any` and
#: `any` is a subtype of nothing, even though both directions are assignable.
#: `join` reads it to pick the wider of two related types — `Circle` with
#: `Shape` is `Shape`, and `int` with `any` is `any` — and `narrow` reads it to
#: downcast a dynamic binding, which is what `_narrow_member` is for.
#:
#: **Nothing here reports a diagnostic.** These are relations, not checks: they
#: answer a type, and the caller owns the span that decides what a surprising
#: one means.
#:
#: ```
#: let u = Universe()
#: equals(u.t_int, u.t_int)                        # true
#: join(u, u.t_int, u.t_never) == u.t_int          # true
#: narrow(u, u.t_any, u.t_str, true) == u.t_str    # true
#: ```
from jaithon.compile.check.assign import is_subtype
from jaithon.compile.check.kinds import JOIN_MAX_MEMBERS, TyKind
from jaithon.compile.check.ty import Type
from jaithon.compile.check.union import union_of
from jaithon.compile.check.universe import Universe

#: Are `a` and `b` the same type?
#:
#: Object identity, and nothing else, because every type comes from
#: `Universe.intern` — which hands back the type it already holds whenever one
#: with the same key exists, so two structurally equal types *are* one object.
#:
#: So the body is `a == b`, as `assign.jai` observes. It is a function anyway
#: because that is the name the C's callers use, and because a named place to
#: state the property is what stops the next reader reaching for a structural
#: comparison: this question is asked on the checker's hottest path, and a tree
#: walk there is the difference between a fast front end and a slow one.
#:
#: Two nulls are equal, matching the C. That is not the claim that two
#: expressions have the same type — it is that neither has one.
#:
#: ```
#: from jaithon.compile.check.universe import Universe, list_of
#: let u = Universe()
#: equals(u.t_int, u.t_int)                            # true
#: equals(list_of(u, u.t_int), list_of(u, u.t_int))    # true
#: equals(u.t_int, u.t_float)                          # false
#: equals(null, null)                                  # true
#: ```
pub fn equals(a: Type?, b: Type?) -> bool {
    return a == b
}

#: The least type that holds both `a` and `b`.
#:
#: What the two arms of an `if` produce, what a list literal's element type is,
#: and what a function with two `return`s returns.
#:
#: **The lattice is the union lattice, not the class hierarchy.** When neither
#: side is a subtype of the other the answer is `a | b`, and it stays `a | b`
#: for two classes with a common ancestor: `Circle` joined with `Square` is
#: `Circle | Square`, never `Shape`. No ancestor search happens here, and adding
#: one would be wrong as well as slow — a superclass is a *lossy* upper bound,
#: so `[a_circle, a_square]` would become a `list[Shape]` and the checker would
#: forget which arm it had. The hierarchy is still used, through the two subtype
#: tests: `Circle` joined with `Shape` is `Shape`, because one really does
#: contain the other.
#:
#: **A union joins by merging.** `union_of` flattens, sorts and deduplicates, so
#: `int | str` joined with `float` is `int | float | str`, and a union joined
#: with one of its own members answers the union unchanged — the subtype test
#: catches that before a second union is built.
#:
#: **Past `JOIN_MAX_MEMBERS` members the answer is `any`.** A least upper bound
#: that keeps growing is worse than no answer: it slows every later comparison
#: and prints a type nobody can read. The test is `>`, so an eight-member union
#: survives and a ninth member collapses the whole thing.
#:
#: The `Never` and `Any` arms are the C's, and neither decides anything the two
#: subtype tests would not: `never` is a subtype of every type and every type is
#: a subtype of `any`, so each arm already falls out of one of the two — a
#: `never` on the left and an `any` on the right out of the first, and the two
#: mirrors of those out of the second.
#: They are kept because they skip two assignability walks on the inputs a fold
#: hits most — `join_all` seeds with `never` — and because the C's order is what
#: this port is measured against, not because deleting them would change an
#: answer. It does not; that was checked pair by pair over the primitives, a
#: class hierarchy, generic parameters, the containers and a union at the cap.
#:
#: ```
#: from jaithon.compile.check.render import render
#: from jaithon.compile.check.union import optional_of
#: let u = Universe()
#: render(join(u, u.t_int, u.t_str))                       # "int | str"
#: join(u, u.t_int, u.t_int) == u.t_int                    # true
#: join(u, u.t_never, u.t_str) == u.t_str                  # true
#: join(u, u.t_any, u.t_str) == u.t_any                    # true
#: render(join(u, u.t_int, optional_of(u, u.t_int)))       # "int?"
#: ```
pub fn join(u: Universe, a: Type?, b: Type?) -> Type {
    if a == null { return b ?? u.t_any }
    if b == null { return a }
    if a == b { return a }

    # `never` inhabits nothing, so it widens nothing; `any` holds everything, so
    # it absorbs. The subtype tests below reach both conclusions on their own —
    # see the note above — so these three lines are a shortcut, not a rule.
    if a.kind == TyKind.Never { return b }
    if b.kind == TyKind.Never { return a }
    if a.kind == TyKind.Any or b.kind == TyKind.Any { return u.t_any }

    # Covers a class and its ancestor, a class and a trait it implements, and
    # `T` with `T?`.
    if is_subtype(u, a, b) { return b }
    if is_subtype(u, b, a) { return a }

    let members: list[Type?] = [a, b]
    let joined = union_of(u, members)
    if joined.kind == TyKind.Union and joined.arg_count() > JOIN_MAX_MEMBERS {
        return u.t_any
    }
    return joined
}

#: The join folded over `types`, starting from `never`.
#:
#: `never` is the identity of the join, so an empty list answers `never` — the
#: type no value has, which is the right answer for a fold over no candidates.
#:
#: Folded rather than collapsed into one `union_of` so that the cap inside
#: `join` fires per step, where the C's fires. Nine unrelated types are what
#: tell the two apart: folded, the ninth trips the cap and the answer is `any`,
#: while one `union_of` caps nothing and hands back a nine-member union.
#:
#: ```
#: from jaithon.compile.check.render import render
#: let u = Universe()
#: render(join_all(u, [u.t_int, u.t_int]))               # "int"
#: render(join_all(u, [u.t_bool, u.t_int, u.t_str]))     # "bool | int | str"
#: ```
pub fn join_all(u: Universe, types: list[Type]) -> Type {
    var result: Type = u.t_never
    for t in types { result = join(u, result, t) }
    return result
}

#: What `t` is once `x is by` has been tested, on the branch `positive` names.
#:
#: **The two directions are not mirror images and must not be made into one.**
#: The positive branch *refines*: a member already at least as specific as `by`
#: survives as itself, and one that is merely wider is downcast to `by`. The
#: negative branch only *subtracts*: a member goes away only when it definitely
#: is a `by`, so `x != null` removes `null` from `int | null` while
#: `x is not Circle` on a `Shape` removes nothing — a shape that is not a circle
#: is still a shape. Subtracting on overlap instead would delete the `Shape` arm
#: and lose every value the test did not rule out.
#:
#: **An empty result is `never`**, in both directions and for a single type as
#: well as a union. The branch cannot be reached, and saying so is what lets the
#: caller report it rather than silently carry a type no value has.
#:
#: Answers null only when `t` is null, matching the C: with no type to narrow
#: there is nothing to say, and a null `by` is a test that says nothing either.
#:
#: ```
#: from jaithon.compile.check.render import render
#: from jaithon.compile.check.union import optional_of
#: let u = Universe()
#: let maybe = optional_of(u, u.t_int)
#: render(narrow(u, maybe, u.t_null, false))      # "int"
#: render(narrow(u, maybe, u.t_null, true))       # "null"
#: render(narrow(u, u.t_any, u.t_str, true))      # "str"
#: render(narrow(u, u.t_int, u.t_str, true))      # "never"
#: render(narrow(u, u.t_int, u.t_str, false))     # "int"
#: ```
pub fn narrow(u: Universe, t: Type?, by: Type?, positive: bool) -> Type? {
    if t == null or by == null { return t }

    if t.kind == TyKind.Union {
        var keep: list[Type?] = []
        for member in t.args {
            if positive {
                let survivor = _narrow_member(u, member, by)
                if survivor != null { keep.push(survivor) }
            } elif not is_subtype(u, member, by) {
                # Subtract, and only what the test definitely removes.
                keep.push(member)
            }
        }
        # No explicit empty case: `union_of` answers `never` for an empty list,
        # which is the right answer here too — nothing survived the test, so
        # the branch cannot be reached. The C spells the case out; it has the
        # same `count <= 0` rule inside `jaiTypeUnion` and so answers the same.
        return union_of(u, keep)
    }

    # No overlap: the branch is unreachable, so the binding has no value.
    if positive { return _narrow_member(u, t, by) ?? u.t_never }
    return if is_subtype(u, t, by) { u.t_never } else { t }
}

# What survives of `m` when the value is known to be a `by`? Either `m` itself,
# already at least as specific, or `by` as a downcast — `x is Circle` where `x`
# is a `Shape` — or nothing, when the test can never succeed for this member.
#
# The order of the two tests is the C's and is observable, because two types can
# each be a subtype of the other without being one type: a generic instantiation
# compares as its declaration, so `Box[int]` and `Box[str]` reach each other,
# and two erased parameters accept every value and so accept each other. Trying
# the keep first means a narrowed binding holds on to the type it had.
#
# A dynamic `m` takes the second test rather than the first, since `any` is a
# subtype of nothing and everything is a subtype of `any`. That is what makes
# `if x is Circle` on an untyped `x` narrow to `Circle` instead of staying
# dynamic inside the branch, and it falls out of the order rather than needing a
# case of its own.
fn _narrow_member(u: Universe, m: Type, by: Type) -> Type? {
    if is_subtype(u, m, by) { return m }
    if is_subtype(u, by, m) { return by }
    return null
}
