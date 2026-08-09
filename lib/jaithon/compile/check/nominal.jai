#: The relations over `TypeDecl`: who inherits from whom, and who declares what.
#:
#: `ty.jai` carries the declaration data and answers only about itself — its own
#: fields, its own members. Every question that has to *walk* lives here,
#: because the walk is the part with rules: an order, a depth cap, and an answer
#: for the declarations this compilation unit cannot see.
#:
#: **Own declarations before inherited ones.** A lookup scans what the
#: declaration itself says, then its traits in the order they were written, then
#: its superclass. That is `declFindMember`'s order in `src/sema/types.c`, and
#: it is what makes an override win over the member it overrides.
#:
#: **Every walk stops at `MAX_DECL_DEPTH`.** A hierarchy is acyclic only because
#: layout rejected the cycles (`E0706`), and these relations are also asked
#: about declarations that are still being laid out, where that has not happened
#: yet. The cap is the C's down to its off-by-one: `declIsSubclassOf` loops
#: while `depth < MAX_DECL_DEPTH` and so tests 32 declarations, while the
#: recursive walks give up on `depth > MAX_DECL_DEPTH` from a first call at
#: depth 0 and so reach 33 levels. `jaiTypeDeclIsSubclassOf` in
#: `src/sema/check.c` has no cap at all; this port gives it one, because a
#: missing answer is recoverable and a hang is not.
#:
#: The walks the C leaves uncapped — the two lookups, the two owners, and
#: `inherits_opaquely` — all take the 33-level shape, and they have to agree on
#: it exactly. A guard that stopped one declaration short of the lookup it
#: excuses would report a name missing on a declaration the lookup had just
#: read, and an owner that stopped short of `find_field` would measure a `prot`
#: field against the wrong class.
#:
#: **What this unit cannot see, it matches by name.** A trait that arrived
#: through the prelude or an import as a runtime object has no `TypeDecl` — only
#: the name the class wrote in its header — so `names_opaque_trait` compares
#: names. That is narrower than "every class satisfies it" and wider than "no
#: class does", and the requirements themselves are enforced at run time.
#: `opaque_super` records the same thing for a superclass, but it deliberately
#: gets no by-name subclass test: the C has none, and an invisible parent is
#: served instead by `inherits_opaquely`, which tells a member lookup to answer
#: `any` rather than claim a name is missing.
from jaithon.compile.check.kinds import MAX_DECL_DEPTH
from jaithon.compile.check.ty import FieldInfo, MemberInfo, Type, TypeDecl

#: True when `sub` is `base`, or descends from it through superclasses alone.
#:
#: Traits are not an axis here, and that is the point. This is the relation
#: assignability asks when the target is a class, and a class-typed slot must
#: not accept a value just because the two happen to name one trait in common.
#: `inherits_from` is the wider question.
#:
#: ```
#: from jaithon.ast import Span
#: let animal = TypeDecl("Animal", Span.none())
#: let dog = TypeDecl("Dog", Span.none())
#: dog.superclass = animal
#: is_subclass_of(dog, animal)      # true
#: is_subclass_of(dog, dog)         # true
#: is_subclass_of(animal, dog)      # false
#: ```
pub fn is_subclass_of(sub: TypeDecl?, base: TypeDecl?) -> bool {
    if sub == null or base == null { return false }
    var current: TypeDecl? = sub
    var depth = 0
    while current != null and depth < MAX_DECL_DEPTH {
        if current == base { return true }
        current = current.superclass
        depth += 1
    }
    return false
}

#: True when `decl` satisfies the trait `wanted`.
#:
#: Reflexive, and closed over both axes: a declaration satisfies a trait it
#: names, a trait its traits extend, and any trait its superclass satisfies.
#:
#: ```
#: from jaithon.ast import Span
#: let comparable = TypeDecl("Comparable", Span.none())
#: let ordered = TypeDecl("Ordered", Span.none())
#: ordered.traits = [comparable]
#: let money = TypeDecl("Money", Span.none())
#: money.traits = [ordered]
#: implements_trait(money, comparable)      # true
#: implements_trait(comparable, money)      # false
#: ```
pub fn implements_trait(decl: TypeDecl?, wanted: TypeDecl?) -> bool {
    if wanted == null { return false }
    return _reaches(decl, wanted, 0)
}

#: True when `decl` sits anywhere in the hierarchy rooted at `ancestor`.
#:
#: The same closure `implements_trait` computes, under the name its other caller
#: uses: a `prot` member is visible to a subclass, and the C asks that through
#: `jaiTypeDeclIsSubclassOf` while it asks trait satisfaction through
#: `declImplements`. Both walk superclasses and traits together and both are
#: reflexive, so they are one relation with two names, and it is spelled twice
#: here only because a caller reading `implements_trait(cls, owner)` about a
#: class would be reading a lie.
pub fn inherits_from(decl: TypeDecl?, ancestor: TypeDecl?) -> bool {
    if ancestor == null { return false }
    return _reaches(decl, ancestor, 0)
}

# Both axes at once. The C reaches this walk from two directions and neither
# passes a depth in, so the cap starts here rather than at the call site.
fn _reaches(decl: TypeDecl?, target: TypeDecl, depth: int) -> bool {
    if decl == null or depth > MAX_DECL_DEPTH { return false }
    if decl == target { return true }
    for extended in decl.traits {
        if _reaches(extended, target, depth + 1) { return true }
    }
    return _reaches(decl.superclass, target, depth + 1)
}

#: True when `decl`, or something it inherits from, named `name` as a trait
#: whose declaration is not visible in this compilation unit.
#:
#: The class still had to write the trait in its header, which is what keeps
#: this from degenerating into "any class will do".
#:
#: ```
#: from jaithon.ast import Span
#: let widget = TypeDecl("Widget", Span.none())
#: widget.opaque_traits = ["Drawable"]
#: names_opaque_trait(widget, "Drawable")     # true
#: names_opaque_trait(widget, "Printable")    # false
#: ```
pub fn names_opaque_trait(decl: TypeDecl?, name: str?) -> bool {
    if name == null { return false }
    return _names_opaque_trait(decl, name, 0)
}

fn _names_opaque_trait(decl: TypeDecl?, name: str, depth: int) -> bool {
    if decl == null or depth > MAX_DECL_DEPTH { return false }
    for written in decl.opaque_traits {
        if written == name { return true }
    }
    for extended in decl.traits {
        if _names_opaque_trait(extended, name, depth + 1) { return true }
    }
    return _names_opaque_trait(decl.superclass, name, depth + 1)
}

#: True when `decl` satisfies the trait type `wanted` — by declaration when
#: there is one, by name when there is not.
#:
#: The single place that choice is made. In the C it is a ternary inside
#: `assignableInto`, written out once for a class or enum reaching a trait and
#: again for a trait reaching a trait; two copies of a rule is two chances for
#: the opaque case to be forgotten in one of them.
pub fn satisfies_trait(decl: TypeDecl?, wanted: Type) -> bool {
    let declared = wanted.decl
    if declared != null { return implements_trait(decl, declared) }
    return names_opaque_trait(decl, wanted.name)
}

#: True when some link of `decl`'s ancestry reached this unit as a runtime
#: object rather than as a declaration.
#:
#: Such a parent's members exist and this unit cannot enumerate them, so it is
#: not entitled to report a name as missing: a failed lookup on one of these
#: answers `any` and leaves the question to the run time. Without it every
#: subclass of an imported class loses its whole inherited surface to `E0410`.
#:
#: The reach is `_find_member`'s and not `is_subclass_of`'s, because this is the
#: answer a failed lookup consults. One declaration shallower and the last link
#: `find_member` reads would be a link this test cannot see, which is `E0410`
#: reported about a parent nobody can enumerate — the exact failure the function
#: exists to prevent.
pub fn inherits_opaquely(decl: TypeDecl?) -> bool {
    return _inherits_opaquely(decl, 0)
}

fn _inherits_opaquely(decl: TypeDecl?, depth: int) -> bool {
    if decl == null or depth > MAX_DECL_DEPTH { return false }
    if decl.opaque_super != null { return true }
    if decl.opaque_traits.len() > 0 { return true }
    return _inherits_opaquely(decl.superclass, depth + 1)
}

#: The field called `name`, taken from `decl` or from an ancestor.
#:
#: The superclass chain is the only axis walked: a trait declares members and
#: never fields, so there is nothing along that one to find. Enum variants are
#: fields, with their tag in `slot`, so this is how a variant is looked up too.
#:
#: `TypeDecl.find_field` asks the same question of a single declaration. The C
#: needs only that much because its layout pass copies a parent's fields down
#: into the child before any lookup runs; nothing has copied anything here, so
#: the inherited answer has to be walked for.
pub fn find_field(decl: TypeDecl?, name: str) -> FieldInfo? {
    return _find_field(decl, name, 0)
}

fn _find_field(decl: TypeDecl?, name: str, depth: int) -> FieldInfo? {
    if decl == null or depth > MAX_DECL_DEPTH { return null }
    let own = decl.find_field(name)
    if own != null { return own }
    return _find_field(decl.superclass, name, depth + 1)
}

#: The member called `name`, taken from `decl`, from a trait it names, or from
#: an ancestor — in that order.
#:
#: The `MemberInfo` comes back rather than its type because a member whose `ty`
#: is null exists but has not been checked yet, and that is a different fact
#: from the member being absent. The C keeps them apart with a `bool` return and
#: an out-parameter that may still be null on success.
pub fn find_member(decl: TypeDecl?, name: str) -> MemberInfo? {
    return _find_member(decl, name, null, 0)
}

#: The getter or the setter called `name`, whichever `want_setter` asks for.
#:
#: A getter and a setter share one name, so a lookup that means the write side
#: has to say so; plain `find_member` answers with whichever was declared first.
#: The C tries this with `want_setter` false before falling back to a plain
#: lookup, because reading is the common case and the getter is the read side.
pub fn find_accessor(decl: TypeDecl?, name: str, want_setter: bool) -> MemberInfo? {
    return _find_member(decl, name, want_setter, 0)
}

fn _find_member(decl: TypeDecl?, name: str, want_setter: bool?, depth: int) -> MemberInfo? {
    if decl == null or depth > MAX_DECL_DEPTH { return null }
    let own = _own_member(decl, name, want_setter)
    if own != null { return own }
    for extended in decl.traits {
        let inherited = _find_member(extended, name, want_setter, depth + 1)
        if inherited != null { return inherited }
    }
    return _find_member(decl.superclass, name, want_setter, depth + 1)
}

# A null `want_setter` is no filter at all, which is the plain lookup.
fn _own_member(decl: TypeDecl, name: str, want_setter: bool?) -> MemberInfo? {
    if want_setter == null { return decl.find_member(name) }
    for member in decl.members {
        if member.name == name and member.is_setter == want_setter { return member }
    }
    return null
}

#: The declaration that first declared the field `name`.
#:
#: The topmost ancestor carrying it, and `decl` itself when no ancestor does —
#: which is also the answer when there is no such field anywhere, matching the
#: C. This is the declaration a visibility diagnostic names, and the one a
#: `prot` field is measured against.
pub fn owner_of_field(decl: TypeDecl?, name: str) -> TypeDecl? {
    if decl == null { return null }
    var owner: TypeDecl = decl
    var current = decl.superclass
    var depth = 0
    while current != null and depth < MAX_DECL_DEPTH {
        if current.find_field(name) != null { owner = current }
        current = current.superclass
        depth += 1
    }
    return owner
}

#: The declaration that first declared the member `name`, on the same terms as
#: `owner_of_field`.
#:
#: A trait is never the answer: only the superclass chain supplies candidates,
#: because a trait's default method belongs to the class that implements it and
#: that is the declaration a diagnostic should name. But whether a link *carries*
#: the name is asked of its traits as well as its own list, because nothing here
#: has copied that default into the implementing class the way the C's layout
#: does. Asking only `TypeDecl.find_member` would make `class Button(Widget)`,
#: where `Widget` implements the trait, report `Button` as the owner of a method
#: `Button` neither wrote nor implements — a `prot` default would then be
#: measured against `Button` and denied to every sibling of it.
pub fn owner_of_member(decl: TypeDecl?, name: str) -> TypeDecl? {
    if decl == null { return null }
    var owner: TypeDecl = decl
    var current = decl.superclass
    var depth = 0
    while current != null and depth < MAX_DECL_DEPTH {
        if _declares_member(current, name, 0) { owner = current }
        current = current.superclass
        depth += 1
    }
    return owner
}

# What one link of the chain would have in its own member list once layout had
# copied its traits' defaults in. The superclass axis is the caller's loop and
# is deliberately not walked again here.
fn _declares_member(decl: TypeDecl?, name: str, depth: int) -> bool {
    if decl == null or depth > MAX_DECL_DEPTH { return false }
    if decl.find_member(name) != null { return true }
    for extended in decl.traits {
        if _declares_member(extended, name, depth + 1) { return true }
    }
    return false
}
