#: The intern table, the primitive singletons, and the type constructors.
#:
#: This is where the port earns back a property the C gets for free. In the C
#: every type lives in one arena and structural equality *is* pointer equality,
#: so `a == b` is a machine word compare and the checker leans on it everywhere.
#: Jaithon has no arena, so identity has to be constructed: every type goes
#: through `Universe.intern`, which returns the existing object whenever one
#: with the same key already exists. Skip the table and two structurally equal
#: types become two objects, every relation over them degrades from a compare
#: into a tree walk, and the checker's hottest path is the one that suffers.
#:
#: The key is `(kind, name, args, ret, fn_flags)` — exactly the C's
#: `typeHash`/`typeMatches` pair. `decl` is deliberately **not** part of it: a
#: named type is identified by kind and name so that a forward reference and
#: the declaration that eventually arrives land on the same object, and the
#: declaration is attached afterwards.
#:
#: `Universe` is an object rather than the C's global `gTypes`, matching
#: `Resolution`'s idiom in `resolve.jai`. That makes the checker testable, and
#: it deletes the C's lifetime hazard outright: there is no `jaiTypesFree` to
#: invalidate every type the process still holds, and no lazy `ensureInit` to
#: forget to call.
#:
#: ```
#: let u = Universe()
#: u.t_int == u.t_int                       # true
#: list_of(u, u.t_int) == list_of(u, u.t_int)   # true, same object
#: u.t_void == u.t_null                     # true, void is a spelling of null
#: ```
from jaithon.compile.check.kinds import TyKind, ordinal
from jaithon.compile.check.ty import Type, TypeDecl

#: Every type in one compilation, and the table that keeps them unique.
pub class Universe {
    #: Interned types in creation order; the enumeration point for tooling.
    var _all: list[Type]
    #: Intern key to type.
    var _table: dict[str, Type]

    pub let t_any: Type
    pub let t_never: Type
    pub let t_null: Type
    pub let t_bool: Type
    pub let t_int: Type
    pub let t_float: Type
    pub let t_str: Type
    pub let t_bytes: Type
    pub let t_range: Type
    #: `void` is not a tenth type. It is `null` under the name a return
    #: position uses, which is why a sema dump records the program's own type
    #: as `null` while a function renders `-> void`.
    pub let t_void: Type

    #: Build a universe with its primitives interned in the C's declaration
    #: order, so their uids are 0 through 8.
    pub fn init(self) {
        self._all = []
        self._table = {}
        self.t_any = self.intern(TyKind.Any)
        self.t_never = self.intern(TyKind.Never)
        self.t_null = self.intern(TyKind.Null)
        self.t_bool = self.intern(TyKind.Bool)
        self.t_int = self.intern(TyKind.Int)
        self.t_float = self.intern(TyKind.Float)
        self.t_str = self.intern(TyKind.Str)
        self.t_bytes = self.intern(TyKind.Bytes)
        self.t_range = self.intern(TyKind.Range)
        self.t_void = self.t_null
    }

    #: The type with this key, creating it only if there is not one already.
    #:
    #: `decl` is applied to the answer whichever branch is taken, because a
    #: named type interned before its declaration was visible has to pick the
    #: declaration up when it arrives. That back-patch is the whole reason
    #: `decl` is the one mutable field on `Type`.
    pub fn intern(
        self,
        kind: TyKind,
        name: str? = null,
        args: list[Type] = [],
        ret: Type? = null,
        fn_flags: int = 0,
        decl: TypeDecl? = null
    ) -> Type {
        let key = _intern_key(kind, name, args, ret, fn_flags)
        if key in self._table {
            let found = self._table[key]
            if decl != null { found.decl = decl }
            return found
        }
        let built = Type(kind, name, args, ret, fn_flags, self._all.len(), decl)
        self._table[key] = built
        self._all.push(built)
        return built
    }

    #: Every interned type, in creation order.
    pub fn all(self) -> list[Type] {
        return self._all
    }

    #: How many distinct types this compilation produced.
    pub fn count(self) -> int {
        return self._all.len()
    }
}

# Args and ret are already interned, so their uids are stable identities and a
# key built from them means exactly what the C's pointer comparison means.
fn _intern_key(kind: TyKind, name: str?, args: list[Type], ret: Type?, fn_flags: int) -> str {
    var parts: list[str] = [f"{ordinal(kind)}", name ?? ""]
    for argument in args {
        parts.push(f"{argument.uid}")
    }
    parts.push(if ret == null { "-" } else { f"{ret.uid}" })
    parts.push(f"{fn_flags}")
    return "\n".join(parts)
}

#: `list[elem]`.
pub fn list_of(u: Universe, elem: Type?) -> Type {
    return u.intern(TyKind.List, null, [elem ?? u.t_any])
}

#: `set[elem]`.
pub fn set_of(u: Universe, elem: Type?) -> Type {
    return u.intern(TyKind.Set, null, [elem ?? u.t_any])
}

#: `dict[key, value]`.
pub fn dict_of(u: Universe, key: Type?, value: Type?) -> Type {
    return u.intern(TyKind.Dict, null, [key ?? u.t_any, value ?? u.t_any])
}

#: `(a, b, ...)`.
pub fn tuple_of(u: Universe, members: list[Type]) -> Type {
    return u.intern(TyKind.Tuple, null, members)
}

#: `fn(params) -> returns`.
#:
#: A null `returns` is kept null rather than defaulted: it means no return type
#: was written, which renders `-> any`, and that is not the same as a written
#: `-> void`.
pub fn fn_of(u: Universe, params: list[Type], returns: Type?, fn_flags: int = 0) -> Type {
    return u.intern(TyKind.Fn, null, params, returns, fn_flags)
}

#: A class, trait or enum type, by name.
pub fn named(u: Universe, kind: TyKind, name: str?, decl: TypeDecl? = null) -> Type {
    return u.intern(kind, name, [], null, 0, decl)
}

#: `T` inside a generic declaration.
pub fn generic_param(u: Universe, name: str?) -> Type {
    return u.intern(TyKind.GenericParam, name)
}

#: A generic instantiation such as `Box[int]`.
pub fn instantiate(u: Universe, base: Type, args: list[Type]) -> Type {
    return u.intern(base.kind, base.name, args, null, 0, base.decl)
}
