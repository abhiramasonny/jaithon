#: The flattened field list a class or enum presents, and the position of a
#: name in it.
#:
#: Superclass fields come first and in declaration order, which is what makes a
#: position an index into the object's storage and an enum variant's ordinal.
#:
#: It is its own file because three unrelated checks index the same list: a
#: pattern destructuring a nominal type, a `match` counting the variants its
#: arms took, and a store into a field. Each would otherwise have to reach into
#: one of the others.
from jaithon.compile.check.kinds import MAX_DECL_DEPTH
from jaithon.compile.check.ty import FieldInfo, TypeDecl

pub fn _layout_fields(decl: TypeDecl?) -> list[FieldInfo] {
    return _collect_fields(decl, 0)
}

fn _collect_fields(decl: TypeDecl?, depth: int) -> list[FieldInfo] {
    if decl is null or depth > MAX_DECL_DEPTH { return [] }
    var out = _collect_fields(decl.superclass, depth + 1)
    for field in decl.fields { out.push(field) }
    return out
}

pub fn _field_index(decl: TypeDecl?, name: str?) -> int {
    if decl is null or name is null { return -1 }
    for (position, field) in _layout_fields(decl).enumerate() {
        if field.name == name { return position }
    }
    return -1
}
