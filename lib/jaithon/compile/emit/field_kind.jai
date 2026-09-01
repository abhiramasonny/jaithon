#: The info bytes the emitter bakes into an instruction from a declared type or
#: a `Visibility`: the code `OP_FIELD_DEF` and `OP_METHOD` carry, the declared
#: kind of a class field, and the packed element kind `OP_ELEM_KIND` puts on a
#: container binding.
#:
#: Grouped because all three are the same move -- a declared type or a
#: `Visibility` in, a byte the VM reads out -- and none of them touches an
#: `Emitter`, so any module of `emit/` may ask them.
from jaithon.ast import TypeExpr, TypeKind, Visibility

pub fn _visibility_code(visibility: Visibility) -> int {
    if visibility == Visibility.Public { return 2 }
    if visibility == Visibility.Protected { return 1 }
    return 0
}

#: The declared kind of a field, for bits 4-7 of `OP_FIELD_DEF`'s info byte.
#: Mirrors `FieldKind` in `src/vm/object/object.h`; 0 is `any`.
#:
#: A HINT, not a guarantee, and the VM guards on it rather than trusting it. So
#: the safe direction is downward: anything this cannot name exactly comes out
#: `any`, which claims nothing. A wrong claim is a false `TypeError` on correct
#: code, which is far worse than a missed one.
#:
#: `T?` is `any` because a nullable holds `null`, and a bare type parameter is
#: `any` because its type is not known here. `list[T]` is still `list`: the
#: container is known even when its element is not.
pub fn _field_kind_code(declared: TypeExpr?, type_params: list[str]) -> int {
    if declared == null { return 0 }
    let t: TypeExpr = declared

    if t.kind == TypeKind.Name {
        let n = t.name ?? ""
        if type_params.contains(n) { return 0 }
        if n == "int" { return 1 }
        if n == "float" { return 2 }
        if n == "bool" { return 3 }
        if n == "str" { return 4 }
        if n == "list" { return 5 }
        if n == "dict" { return 6 }
        if n == "any" { return 0 }
        #: A user type name -- a class, an enum, a trait, or an alias for one.
        #: Everything the language can put behind such a name is a heap object,
        #: which is all code 8 claims. It is a hint for the compiled tier, which
        #: guards it; the run-time check accepts everything for it, because
        #: tightening a check that has always passed would reject code that
        #: works today.
        #:
        #: A TYPE PARAMETER is excluded above and must stay excluded: `T` can be
        #: an int.
        return 8
    }

    if t.kind == TypeKind.Generic {
        let n = t.name ?? ""
        if n == "list" { return 5 }
        if n == "dict" { return 6 }
        return 0
    }

    return 0
}

#: The packed `OP_ELEM_KIND` byte for a declared container type, or 0 when the
#: type is not a container or its element type cannot be named.
#:
#: Low nibble is the element (or dict value) kind, high nibble the dict key
#: kind, both `FieldKind` codes. Zero means "promise nothing", which is what an
#: untyped container carries and what makes the guard one compare on the common
#: path.
pub fn _container_kind_code(declared: TypeExpr?) -> int {
    if declared == null { return 0 }
    let t: TypeExpr = declared
    if t.kind != TypeKind.Generic { return 0 }

    let name = t.name ?? ""
    if name == "list" {
        if t.args.len() != 1 { return 0 }
        return _field_kind_code(t.args[0], []) & 0xF
    }
    if name == "dict" {
        if t.args.len() != 2 { return 0 }
        let key = _field_kind_code(t.args[0], []) & 0xF
        let value = _field_kind_code(t.args[1], []) & 0xF
        return key << 4 | value
    }
    return 0
}
