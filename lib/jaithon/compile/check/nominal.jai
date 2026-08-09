#: Relations over `TypeDecl`: who inherits from whom, and who declares what.
#: Every walk caps at `MAX_DECL_DEPTH`: a hierarchy is acyclic only once layout
#: has rejected the cycles (`E0706`), and these are asked during layout. The
#: `while` loops reach 32 declarations and the recursive walks 33; the recursive
#: walks must all agree on 33, or a guard stops short of the lookup it excuses.
from jaithon.compile.check.kinds import MAX_DECL_DEPTH
from jaithon.compile.check.ty import FieldInfo, MemberInfo, Type, TypeDecl

#: True when `sub` is `base`, or descends from it through superclasses alone.
#: Traits are not an axis: this is the relation assignability asks for a class.
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

#: True when `decl` satisfies the trait `wanted`, over traits and superclasses.
pub fn implements_trait(decl: TypeDecl?, wanted: TypeDecl?) -> bool {
    if wanted == null { return false }
    return _reaches(decl, wanted, 0)
}

#: True when `decl` is anywhere under `ancestor`: `implements_trait`, renamed.
pub fn inherits_from(decl: TypeDecl?, ancestor: TypeDecl?) -> bool {
    if ancestor == null { return false }
    return _reaches(decl, ancestor, 0)
}

fn _reaches(decl: TypeDecl?, target: TypeDecl, depth: int) -> bool {
    if decl == null or depth > MAX_DECL_DEPTH { return false }
    if decl == target { return true }
    for extended in decl.traits {
        if _reaches(extended, target, depth + 1) { return true }
    }
    return _reaches(decl.superclass, target, depth + 1)
}

#: True when `decl` or an ancestor named `name` as a trait this unit cannot see.
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

#: True when `decl` satisfies `wanted`: by its declaration, or by name if none.
pub fn satisfies_trait(decl: TypeDecl?, wanted: Type) -> bool {
    let declared = wanted.decl
    if declared != null { return implements_trait(decl, declared) }
    return names_opaque_trait(decl, wanted.name)
}

#: True when a link of `decl`'s ancestry reached this unit as a runtime object
#: rather than a declaration; a failed lookup then answers `any`, not `E0410`.
pub fn inherits_opaquely(decl: TypeDecl?) -> bool {
    return _inherits_opaquely(decl, 0)
}

fn _inherits_opaquely(decl: TypeDecl?, depth: int) -> bool {
    if decl == null or depth > MAX_DECL_DEPTH { return false }
    if decl.opaque_super != null { return true }
    if decl.opaque_traits.len() > 0 { return true }
    return _inherits_opaquely(decl.superclass, depth + 1)
}

#: The field called `name`, from `decl` or an ancestor. Superclasses are the
#: only axis, traits declare no fields; enum variants are fields, tag in `slot`.
pub fn find_field(decl: TypeDecl?, name: str) -> FieldInfo? {
    return _find_field(decl, name, 0)
}

fn _find_field(decl: TypeDecl?, name: str, depth: int) -> FieldInfo? {
    if decl == null or depth > MAX_DECL_DEPTH { return null }
    let own = decl.find_field(name)
    if own != null { return own }
    return _find_field(decl.superclass, name, depth + 1)
}

#: The member called `name`, from `decl`, its traits, then an ancestor, per
#: `declFindMember` in `src/sema/types.c`. A null `ty` is unchecked, not absent.
pub fn find_member(decl: TypeDecl?, name: str) -> MemberInfo? {
    return _find_member(decl, name, null, 0)
}

#: The getter or the setter called `name`, whichever `want_setter` asks for. The
#: two share one name, so plain `find_member` answers with whichever came first.
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

fn _own_member(decl: TypeDecl, name: str, want_setter: bool?) -> MemberInfo? {
    if want_setter == null { return decl.find_member(name) }
    for member in decl.members {
        if member.name == name and member.is_setter == want_setter { return member }
    }
    return null
}

#: The topmost ancestor declaring the field `name`, and `decl` itself when none
#: does or there is no such field, matching the C. `prot` is measured against it.
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

#: The first declaration of the member `name`, on `owner_of_field`'s terms. A
#: trait is never the answer, but a link counts if one of its traits carries the
#: name, or `Button(Widget)` is named owner of a default `Widget` implements.
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

fn _declares_member(decl: TypeDecl?, name: str, depth: int) -> bool {
    if decl == null or depth > MAX_DECL_DEPTH { return false }
    if decl.find_member(name) != null { return true }
    for extended in decl.traits {
        if _declares_member(extended, name, depth + 1) { return true }
    }
    return false
}
