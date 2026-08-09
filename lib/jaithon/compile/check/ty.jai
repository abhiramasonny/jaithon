#: The semantic type, and the declaration data a nominal type carries.
#:
#: Data only. Every relation over these — assignability, join, narrowing,
#: rendering — lives in its own file, so this one can be imported from anywhere
#: in `jaithon.compile.check` without a cycle.
#:
#: `Type` deliberately defines **no `__eq__` and no `__hash__`**. Two types are
#: the same type when they are the same object, which is what the intern table
#: in `universe.jai` exists to guarantee; the C gets the same property free from
#: an arena. Defining structural equality here would make `a == b` quietly walk
#: two trees, and every assignability check in the checker is a comparison, so
#: the walk would land on the hottest path there is.
#:
#: It defines no `__str__` either. Rendering is `render.jai`, whose output is a
#: byte-exact contract with the C, and a second rendering path is exactly how
#: that contract gets lost.
from jaithon.ast import Node, Span
from jaithon.compile.check.kinds import TyKind, ordinal

#: One semantic type.
#:
#: Built only by `Universe.intern`, never directly: a `Type` that did not come
#: from the intern table compares unequal to its own twin and breaks every
#: relation that assumes identity.
#:
#: `args` and `ret` mean different things per kind, exactly as in the C:
#:
#: - `List`, `Set` — `args[0]` is the element
#: - `Dict` — `args[0]` is the key, `args[1]` the value
#: - `Tuple`, `Union` — `args` are the members
#: - `Fn` — `args` are the parameters, `ret` the return type
#: - `Class`, `Trait`, `Enum`, `GenericParam` — `name`, and `args` when generic
pub class Type {
    #: What this type is.
    pub let kind: TyKind
    #: `ordinal(kind)`, cached because the union sort reads it per comparison.
    pub let ord: int
    #: The identifier of a named type, null otherwise.
    pub let name: str?
    #: Element, key/value, members or parameters, depending on `kind`.
    pub let args: list[Type]
    #: A function's return type. Null means the function had none written.
    pub let ret: Type?
    #: `FN_VARIADIC` and `FN_KWREST`, or zero.
    pub let fn_flags: int
    #: Position in the universe's intern order. Stable within one compilation
    #: and useful only for debugging; identity is the object, not this.
    pub let uid: int

    #: The declaration this named type resolves to, once one is visible.
    #:
    #: The one mutable field, and it has to be. A named type is interned by
    #: kind and name so that a forward reference and the eventual declaration
    #: land on the same object; the declaration is therefore attached after the
    #: type already exists. `decl` is for that reason **not** part of the
    #: intern key.
    pub var decl: TypeDecl?

    #: Intern a type. Call `Universe.intern`, not this.
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

    #: How many arguments this type carries.
    pub fn arg_count(self) -> int {
        return self.args.len()
    }

    #: True when this type is of `kind`.
    pub fn is_kind(self, kind: TyKind) -> bool {
        return self.kind == kind
    }

    #: True when this type takes part in a nominal hierarchy.
    pub fn is_named(self) -> bool {
        return self.kind == TyKind.Class or self.kind == TyKind.Trait or self.kind == TyKind.Enum
    }

    #: True when the function this describes collects extra positionals.
    pub fn is_variadic(self) -> bool {
        return self.fn_flags & 1 != 0
    }

    #: True when the function this describes collects keyword arguments.
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

#: What a class, trait or enum declaration says about itself.
#:
#: Lives here rather than beside the checker because the nominal relations are
#: its primary consumer, and putting it there would make this file and that one
#: mutually dependent — which is exactly what the C does, with `types.c`
#: reaching into `check.h`.
pub class TypeDecl {
    pub let name: str
    pub let span: Span
    #: The declaration node, or null for a type known only by name.
    pub var node: Node?
    #: The interned type this declares.
    pub var ty: Type?
    pub var superclass: TypeDecl?
    pub var traits: list[TypeDecl]
    #: Parallel to `traits`: the arguments each was written with, so a
    #: requirement can be substituted before it is compared against what
    #: implements it. Empty for a non-generic trait.
    pub var trait_bindings: list[list[Type]]

    #: Traits named in the header whose declaration this compilation unit
    #: cannot see — they arrived through the prelude or an import as a runtime
    #: object. Only the name is known, so a class implements one of these
    #: exactly when it named it, and the requirements are enforced at run time.
    pub var opaque_traits: list[str]
    #: The superclass named in the header when its declaration is not visible
    #: either. Without it a subclass of an imported class looks like a class
    #: with no parent and every inherited member reads as undefined.
    pub var opaque_super: str?

    pub var fields: list[FieldInfo]
    pub var members: list[MemberInfo]

    pub var is_trait: bool
    pub var is_enum: bool
    pub var is_abstract: bool
    #: How many variants an enum declares; zero otherwise.
    pub var variant_count: int

    pub fn init(self, name: str, span: Span, node: Node? = null) {
        self.name = name
        self.span = span
        self.node = node
        self.ty = null
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

    #: The field called `name`, or null.
    pub fn find_field(self, name: str) -> FieldInfo? {
        for field in self.fields {
            if field.name == name { return field }
        }
        return null
    }

    #: The member called `name`, or null.
    pub fn find_member(self, name: str) -> MemberInfo? {
        for member in self.members {
            if member.name == name { return member }
        }
        return null
    }
}
