#: `TypeExpr` -- a type as it was written, before the checker resolves it --
#: and the `TypeKind` that tags it.
#:
#: `TypeKind` sits here rather than with the node kinds because it tags a
#: `TypeExpr` and nothing else; no field schema and no group predicate reads
#: it.
#:
#: `to_source` stays a method instead of joining the rest of the source printer
#: in `ast_unparse.jai`, which calls it and adds nothing: a type expression
#: prints from itself alone. Its parenthesisation of an optional wrapping a
#: function or a union is load-bearing for `fmt`, and the reason is recorded at
#: that branch rather than up here.
from jaithon.ast.span import Span

#: The shape of a type expression, mirroring the C `AstTypeKind`.
#: `Const` is an integer literal in type position (`Tensor[float, 2, 3]`).
pub enum TypeKind { Name, Generic, Optional, Union, Fn, Tuple, Infer, Const }

#: A type as it was written, before the checker resolves it.
#:
#: Mirrors the C `AstType`: `args` holds a generic's arguments, a union's
#: members, a tuple's members or a function's parameters, and `inner` holds an
#: optional's element or a function's return type.
#:
#: ```
#: TypeExpr.generic("list", [TypeExpr.named("int")]).to_source()   # "list[int]"
#: TypeExpr.optional(TypeExpr.named("str")).to_source()            # "str?"
#: ```
pub class TypeExpr {
    #: Which kind of type expression this is.
    pub let kind: TypeKind
    #: The identifier of a `Name` or `Generic` type, null otherwise.
    pub let name: str?
    #: Generic arguments, union members, tuple members or function parameters.
    pub let args: list[TypeExpr]
    #: An optional's element or a function's return type, null otherwise.
    pub let inner: TypeExpr?
    #: Where the type was written.
    pub let span: Span

    #: Create a type expression of `kind`; prefer the named constructors below.
    pub fn init(
        self,
        kind: TypeKind,
        name: str? = null,
        args: list[TypeExpr] = [],
        inner: TypeExpr? = null,
        span: Span? = null
    ) {
        self.kind = kind
        self.name = name
        self.args = args
        self.inner = inner
        self.span = span ?? Span.none()
    }

    #: A plain named type, such as `int` or `Account`.
    pub static fn named(name: str) -> TypeExpr {
        return TypeExpr(TypeKind.Name, name: name)
    }

    #: An applied generic, such as `dict[str, int]`.
    pub static fn generic(name: str, args: list[TypeExpr]) -> TypeExpr {
        if args.len() == 0 { throw ValueError(f"generic type {name} needs at least one argument") }
        return TypeExpr(TypeKind.Generic, name: name, args: args)
    }

    #: `inner?`, the union of `inner` and null.
    pub static fn optional(inner: TypeExpr) -> TypeExpr {
        return TypeExpr(TypeKind.Optional, inner: inner)
    }

    #: `A | B | ...`, flattened exactly as the parser flattens it.
    pub static fn union_of(members: list[TypeExpr]) -> TypeExpr {
        if members.len() < 2 { throw ValueError("a union type needs at least two members") }
        return TypeExpr(TypeKind.Union, args: members)
    }

    #: `fn(params) -> returns`.
    pub static fn function(params: list[TypeExpr], returns: TypeExpr) -> TypeExpr {
        return TypeExpr(TypeKind.Fn, args: params, inner: returns)
    }

    #: `(A, B)`.
    pub static fn tuple_of(members: list[TypeExpr]) -> TypeExpr {
        return TypeExpr(TypeKind.Tuple, args: members)
    }

    #: The absent annotation, which the checker fills in.
    pub static fn infer() -> TypeExpr {
        return TypeExpr(TypeKind.Infer)
    }

    #: Render this type as source. `TypeKind.Infer` renders as the empty string.
    pub fn to_source(self) -> str {
        match self.kind {
            TypeKind.Name => { return self.name ?? "_" }
            TypeKind.Generic => {
                return (self.name ?? "_") + "[" + ", ".join(self._arg_sources()) + "]"
            }
            TypeKind.Optional => {
                let inner = self.inner
                if inner is null { return "null" }
                #: `?` binds tighter than a function type's arrow and than a
                #: union's bar, so an optional wrapping either of those has to
                #: keep the parentheses the source wrote. Without them
                #: `(fn(float) -> float)?` printed as `fn(float) -> float?`,
                #: which is a function RETURNING an optional -- a different
                #: type that happens to be spelt with the same characters.
                #: `fmt` rewrites a file through this printer, so the file came
                #: back not merely reformatted but broken: the checker reported
                #: "expected `fn(float) -> float?`, found `null`" on a default
                #: argument that had been correct a moment earlier.
                if inner.kind == TypeKind.Fn or inner.kind == TypeKind.Union {
                    return "(" + inner.to_source() + ")?"
                }
                return inner.to_source() + "?"
            }
            TypeKind.Union => { return " | ".join(self._arg_sources()) }
            TypeKind.Fn => {
                let head = "fn(" + ", ".join(self._arg_sources()) + ") -> "
                let returns = self.inner
                if returns is null { return head + "void" }
                return head + returns.to_source()
            }
            TypeKind.Tuple => { return "(" + ", ".join(self._arg_sources()) + ")" }
            TypeKind.Infer => { return "" }
            TypeKind.Const => { return self.name ?? "0" }
        }
    }

    fn _arg_sources(self) -> list[str] {
        return [argument.to_source() for argument in self.args]
    }

    fn __eq__(self, other: TypeExpr) -> bool {
        return self.kind == other.kind and
            self.name == other.name and
            self.args == other.args and
            self.inner == other.inner and
            self.span == other.span
    }

    fn __str__(self) -> str {
        return self.to_source()
    }
}
