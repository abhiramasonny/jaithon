#: The intern table, the primitive singletons, and the type constructors.
from jaithon.compile.check.kinds import TyKind, ordinal
from jaithon.compile.check.ty import Type, TypeDecl

#: Every type in one compilation, and the table that keeps them unique.
pub class Universe {
    var _all: list[Type]
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
    #: `void` is `null` under the name a return position uses; dumps say `null`.
    pub let t_void: Type

    #: Primitives are interned in the C's declaration order, so uids are 0 to 8.
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

    #: The type with this key, or the existing one. `decl` is outside the key,
    #: so a forward reference and its later declaration land on one object.
    pub fn intern(
        self,
        kind: TyKind,
        name: str? = null,
        args: list[Type] = [],
        ret: Type? = null,
        fn_flags: int = 0,
        decl: TypeDecl? = null
    ) -> Type {
        var owned: list[Type] = []
        for argument in args { owned.push(self._adopt(argument)) }
        let owned_ret = if ret == null { null } else { self._adopt(ret) }

        let key = _intern_key(kind, name, owned, owned_ret, fn_flags)
        if key in self._table {
            let found = self._table[key]
            if decl != null { found.decl = decl }
            return found
        }
        let built = Type(kind, name, owned, owned_ret, fn_flags, self._all.len(), decl)
        self._table[key] = built
        self._all.push(built)
        return built
    }

    #: This universe's equivalent of `t`, interning it if it is a stranger.
    #: The key is made of uids and a hand-built `Type` has `uid == -1`, so
    #: unadopted they collide: `list[list[str]]` rendered `list[set[bool]]`.
    fn _adopt(self, t: Type) -> Type {
        if t.uid >= 0 { return t }
        return self.intern(t.kind, t.name, t.args, t.ret, t.fn_flags, t.decl)
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

#: `fn(params) -> returns`. An unwritten return means `any` (spec §6) and is
#: defaulted here, so `fn(int)` and `fn(int) -> any` are one interned type.
pub fn fn_of(u: Universe, params: list[Type], returns: Type?, fn_flags: int = 0) -> Type {
    return u.intern(TyKind.Fn, null, params, returns ?? u.t_any, fn_flags)
}

#: A class, trait or enum type by name; a nameless one answers `any`.
pub fn named(u: Universe, kind: TyKind, name: str?, decl: TypeDecl? = null) -> Type {
    if name == null { return u.t_any }
    return u.intern(kind, name, [], null, 0, decl)
}

#: `T` inside a generic declaration.
pub fn generic_param(u: Universe, name: str?) -> Type {
    return named(u, TyKind.GenericParam, name)
}

#: A generic instantiation such as `Box[int]`.
pub fn instantiate(u: Universe, base: Type, args: list[Type]) -> Type {
    return u.intern(base.kind, base.name, args, null, 0, base.decl)
}
