#: The semantic type, and the declaration data a nominal type carries. Data
#: only, so this file imports from anywhere in `jaithon.compile.check` without
#: a cycle. `Type` defines no `__eq__`, `__hash__` or `__str__`: type identity
#: is object identity via the intern table in `universe.jai`, and rendering is
#: `render.jai`, byte-exact against the C.
from jaithon.ast import Node, Span
from jaithon.compile.check.kinds import TyKind, ordinal

#: One semantic type, built only by `Universe.intern`: a `Type` from anywhere
#: else compares unequal to its twin and breaks every relation. `args` is the
#: element for `List`/`Set`, key then value for `Dict`, `Tuple`/`Union` members,
#: a `Fn`'s parameters with `ret` its return, or a generic's arguments.
pub class Type {
    pub let kind: TyKind
    #: `ordinal(kind)`, cached because the union sort reads it per comparison.
    pub let ord: int
    pub let name: str?
    pub let args: list[Type]
    pub let ret: Type?
    #: `FN_VARIADIC` and `FN_KWREST`, or zero.
    pub let fn_flags: int
    pub let uid: int

    #: The declaration this named type resolves to. Named types intern by kind
    #: and name, so `decl` is attached after interning and is not in the key.
    pub var decl: TypeDecl?

    pub fn init(
        self,
        kind: TyKind,
        name: str? = null,
        args: list[Type] = [],
        ret: Type? = null,
        fn_flags: int = 0,
        uid: int = -1,
        decl: TypeDecl? = null
    ) {
        self.kind = kind
        self.ord = ordinal(kind)
        self.name = name
        self.args = args
        self.ret = ret
        self.fn_flags = fn_flags
        self.uid = uid
        self.decl = decl
    }

    pub fn arg_count(self) -> int {
        return self.args.len()
    }

    pub fn is_kind(self, kind: TyKind) -> bool {
        return self.kind == kind
    }

    pub fn is_named(self) -> bool {
        return self.kind == TyKind.Class or self.kind == TyKind.Trait or self.kind == TyKind.Enum
    }

    pub fn is_variadic(self) -> bool {
        return self.fn_flags & 1 != 0
    }

    pub fn has_kwrest(self) -> bool {
        return self.fn_flags & 2 != 0
    }
}

#: One field of a class, laid out.
pub class FieldInfo {
    pub let name: str
    pub var ty: Type?
    #: Slot in the instance, parents first.
    pub var slot: int
    pub var visibility: int
    pub var is_static: bool
    pub var is_let: bool
    pub var has_default: bool

    pub fn init(
        self,
        name: str,
        ty: Type? = null,
        slot: int = -1,
        visibility: int = 0,
        is_static: bool = false,
        is_let: bool = false,
        has_default: bool = false
    ) {
        self.name = name
        self.ty = ty
        self.slot = slot
        self.visibility = visibility
        self.is_static = is_static
        self.is_let = is_let
        self.has_default = has_default
    }
}

#: One method, getter or setter of a class or trait.
pub class MemberInfo {
    pub let name: str
    #: Always a `Fn` type.
    pub var ty: Type?
    pub var visibility: int
    pub var is_static: bool
    pub var is_getter: bool
    pub var is_setter: bool
    pub var is_abstract: bool
    pub var node: Node?

    pub fn init(
        self,
        name: str,
        ty: Type? = null,
        visibility: int = 0,
        is_static: bool = false,
        is_getter: bool = false,
        is_setter: bool = false,
        is_abstract: bool = false,
        node: Node? = null
    ) {
        self.name = name
        self.ty = ty
        self.visibility = visibility
        self.is_static = is_static
        self.is_getter = is_getter
        self.is_setter = is_setter
        self.is_abstract = is_abstract
        self.node = node
    }
}

#: What a class, trait or enum declaration says about itself. It lives here and
#: not beside the checker, which would make the two files mutually dependent.
pub class TypeDecl {
    pub let name: str
    pub let span: Span
    pub var node: Node?
    pub var ty: Type?
    #: The module this was read out of, null when the checked file declares it.
    #: Held as `any` because naming `check.modsig.ModuleSig` closes an import
    #: cycle. `TypeDecl.origin` in `src/sema/check.h`: annotations inside are
    #: in that module's namespace, so laying it out needs that scope installed.
    pub var origin: any?
    pub var superclass: TypeDecl?
    pub var traits: list[TypeDecl]
    #: Parallel to `traits`: the arguments each was written with, so a
    #: requirement is substituted before comparison. Empty if non-generic.
    pub var trait_bindings: list[list[Type]]

    #: Traits named in the header whose declaration this unit cannot see: a
    #: class implements one by naming it, and requirements hold at run time.
    pub var opaque_traits: list[str]
    #: Likewise for the superclass. Without it a subclass of an imported class
    #: looks parentless and every inherited member reads as undefined.
    pub var opaque_super: str?

    pub var fields: list[FieldInfo]
    pub var members: list[MemberInfo]

    pub var is_trait: bool
    pub var is_enum: bool
    pub var is_abstract: bool
    pub var variant_count: int

    pub fn init(self, name: str, span: Span, node: Node? = null) {
        self.name = name
        self.span = span
        self.node = node
        self.ty = null
        self.origin = null
        self.superclass = null
        self.traits = []
        self.trait_bindings = []
        self.opaque_traits = []
        self.opaque_super = null
        self.fields = []
        self.members = []
        self.is_trait = false
        self.is_enum = false
        self.is_abstract = false
        self.variant_count = 0
    }

    pub fn find_field(self, name: str) -> FieldInfo? {
        for field in self.fields {
            if field.name == name { return field }
        }
        return null
    }

    pub fn find_member(self, name: str) -> MemberInfo? {
        for member in self.members {
            if member.name == name { return member }
        }
        return null
    }
}
