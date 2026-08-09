#: The checker: what turns a resolved tree into a typed one.
#:
#: A port of the parts of `src/sema/check.c` the other files in this directory
#: leave to it — the bridge from a written annotation to a semantic type, the
#: type-name scope that bridge resolves against, and the program-level passes
#: that register declarations before anything is checked. Everything else in
#: `check.c` already lives beside the code that needs it: `require_assignable`
#: and the cast insertion in `expr.jai`, narrowing in `stmt.jai`, layout in
#: `decl.jai`.
#:
#: This file is what closes the `Ctx` seam. `expr.jai` declares thirteen
#: callbacks with harmless defaults; `stmt.install` and `decl.install` supply
#: nine of them and `install` here supplies the rest, so a `Ctx` built by
#: `check_program` has no silent no-op left in it.
#:
#: **`resolve_ast_type` is the one that matters.** Until it is installed every
#: written annotation answers `any`, and an `any` on both sides of an
#: assignment needs neither a runtime guard nor a widening — so a checker
#: without it emits none of the `OP_TYPE_GUARD` and `OP_TO_FLOAT` that account
#: for roughly half the bytecode the two front ends disagree about. It is the
#: difference between a checker that runs and a checker that does anything.
from jaithon.ast import Node, NodeKind, TypeExpr, TypeKind, kind_name
from jaithon.compile.check.decl import layout_decl, register_decl
from jaithon.compile.check.dump import Record
from jaithon.compile.check.expr import Ctx
from jaithon.compile.check.render import render
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.union import optional_of, union_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, list_of
from jaithon.compile.check.universe import named, set_of, tuple_of
from jaithon.compile.lexer import DiagnosticBag
from jaithon.compile.resolve import Resolution
import jaithon.compile.check.decl as decl_mod
import jaithon.compile.check.stmt as stmt_mod

#: An annotation named a type nothing declares.
pub const E0402_UNKNOWN_TYPE: str = "E0402"

#: A generic was written with the wrong number of arguments.
pub const E0407_GENERIC_ARITY: str = "E0407"

#: The names that are types without anything declaring them.
#:
#: `void` is deliberately absent as a distinct entry: it resolves to `null`,
#: because the two are one type and `void` is only the spelling a return
#: position uses.
#:
#: The bare container names are the fully dynamic form of the generic — `list`
#: is `list[any]`. `tuple` is not: its arity is part of the type, so "a tuple of
#: any shape" is not in the lattice at all and `any` is the honest answer.
fn _builtin_scalar(u: Universe, name: str?) -> Type? {
    if name == null { return null }
    match name {
        "any" => { return u.t_any }
        "void" => { return u.t_void }
        "null" => { return u.t_null }
        "bool" => { return u.t_bool }
        "int" => { return u.t_int }
        "float" => { return u.t_float }
        "str" => { return u.t_str }
        "bytes" => { return u.t_bytes }
        "range" => { return u.t_range }
        "never" => { return u.t_never }
        "list" => { return list_of(u, u.t_any) }
        "set" => { return set_of(u, u.t_any) }
        "dict" => { return dict_of(u, u.t_any, u.t_any) }
        "tuple" => { return u.t_any }
        _ => { return null }
    }
}

#: Classes the runtime provides that no source file declares.
pub let BUILTIN_CLASSES: list[str] = [
    "Error",
    "Exception",
    "TypeError",
    "ValueError",
    "IndexError",
    "KeyError",
    "AttributeError",
    "NameError",
    "ZeroDivisionError",
    "OverflowError",
    "RuntimeError",
    "NotImplementedError",
    "AssertionError",
    "StopIteration",
    "ParseError",
    "ImportError",
    "IOError",
    "FileNotFoundError",
    "PermissionError",
    "OSError",
    "TimeoutError",
    "LookupError",
    "ArithmeticError",
    "RecursionError",
    "UnicodeError",
]

fn _builtin_class(u: Universe, name: str?) -> Type? {
    if name == null { return null }
    if name in BUILTIN_CLASSES { return named(u, TyKind.Class, name) }
    return null
}

#: The type a bare name means here, or null when nothing names one.
#:
#: Innermost binding wins: a generic parameter shadows a class of the same
#: name, which shadows a module-level alias. That stack is `Ctx.type_names`,
#: pushed by `decl.push_generics` and `decl.push_self`.
pub fn lookup_type_name(c: Ctx, name: str?) -> Type? {
    if name == null { return null }

    let scoped = c.scoped_type_name(name)
    if scoped != null { return scoped }

    let scalar = _builtin_scalar(c.universe, name)
    if scalar != null { return scalar }

    let declared = c.decl_named(name)
    if declared != null and declared.ty != null { return declared.ty }

    # The C has one more branch here and this port cannot reach it: it asks the
    # resolver for the symbol `name` binds and takes the type off it, which is
    # how an imported class or a runtime-registered builtin resolves. Jaithon's
    # `Resolution` binds symbols to *nodes* (`symbol_of`) and exposes no
    # lookup by bare name, so there is nothing to ask. The consequence is that
    # an imported type used in an annotation falls through to the builtin-class
    # table and then to null, and the caller reports E0402 where the C
    # silently answers `any`. Phase 5 owns the module graph and is where the
    # lookup that closes this belongs.
    return _builtin_class(c.universe, name)
}

# `list`, `set`, `dict` and `tuple` written with arguments. Answers whether it
# handled the name at all, so a user generic falls through to the erasure path.
fn _builtin_generic(c: Ctx, t: TypeExpr) -> (bool, Type) {
    let name = t.name
    var want = -1
    if name == "list" or name == "set" {
        want = 1
    } elif name == "dict" {
        want = 2
    } elif name == "tuple" {
        want = -2
    } else {
        return (false, c.universe.t_any)
    }

    let given = t.args.len()
    if want >= 0 and given != want {
        let plural = if want == 1 { "" } else { "s" }
        let was_were = if given == 1 { "was" } else { "were" }
        let d = c.error(
            E0407_GENERIC_ARITY,
            t.span,
            f"`{name}` takes {want} type argument{plural} but {given} {was_were} given"
        )
        let shape = if want == 1 { "T" } else { "K, V" }
        d.help = f"write `{name}[{shape}]`"
        return (true, c.universe.t_any)
    }

    if want == 1 {
        let elem = resolve_ast_type(c, t.args[0])
        return (true, if name == "list" {
            list_of(c.universe, elem)
        } else {
            set_of(c.universe, elem)
        })
    }
    if want == 2 {
        return (
            true,
            dict_of(
                c.universe,
                resolve_ast_type(c, t.args[0]),
                resolve_ast_type(c, t.args[1])
            )
        )
    }

    var members: list[Type] = []
    for argument in t.args { members.push(resolve_ast_type(c, argument) ?? c.universe.t_any) }
    return (true, tuple_of(c.universe, members))
}

# How many type parameters a declaration takes.
fn _generic_arity(d: TypeDecl?) -> int {
    if d == null { return 0 }
    let node = d.node
    if node == null { return 0 }
    if "generics" not in node.fields { return 0 }
    let generics = node.fields["generics"]
    if generics is list { return generics.len() }
    return 0
}

#: The semantic type a written annotation means.
#:
#: The bridge between `jaithon.ast.TypeExpr` — how a type was *written*, seven
#: syntactic shapes — and `check.ty.Type`, what it *is*. Every guard and every
#: widening the checker inserts starts here, because both are decided by
#: comparing a value's type against an annotation's, and without this the
#: annotation side is always `any`.
#:
#: Results are memoised per `TypeExpr` in `Ctx`, which is the C's
#: `AstType.resolved` field; a Jaithon `TypeExpr` has no such field, so the
#: cache is a side table like every other annotation this port carries.
#:
#: **Generics are erased.** `Box[int]` checks its arity and then stands for the
#: bare `Box`. Resolving it to an instantiation instead would be defensible in
#: isolation and wrong here: the C never builds one, so `make sema-diff` would
#: report `Box[int]` against `Box` on every generic annotation in the tree.
pub fn resolve_ast_type(c: Ctx, t: TypeExpr?) -> Type? {
    if t == null { return c.universe.t_any }

    let cached = c.resolved_type(t)
    if cached != null { return cached }

    var result: Type? = c.universe.t_any
    match t.kind {
        TypeKind.Infer => { result = c.universe.t_any }
        TypeKind.Name => {
            var named_type = lookup_type_name(c, t.name)
            if named_type == null {
                c.error(E0402_UNKNOWN_TYPE, t.span, f"unknown type `{t.name ?? "?"}`")
                named_type = c.universe.t_any
            }
            result = named_type
        }
        TypeKind.Generic => {
            let handled = _builtin_generic(c, t)
            if handled[0] {
                result = handled[1]
            } else {
                # The arguments are resolved for their own diagnostics even
                # though erasure throws them away.
                for argument in t.args { let _ = resolve_ast_type(c, argument) }
                let base = lookup_type_name(c, t.name)
                if base == null {
                    c.error(E0402_UNKNOWN_TYPE, t.span, f"unknown type `{t.name ?? "?"}`")
                    result = c.universe.t_any
                } else {
                    let want = _generic_arity(base.decl)
                    let given = t.args.len()
                    if base.decl != null and want != given {
                        let plural = if want == 1 { "" } else { "s" }
                        let was_were = if given == 1 { "was" } else { "were" }
                        c.error(
                            E0407_GENERIC_ARITY,
                            t.span,
                            f"`{t.name}` takes {want} type argument{plural} but " +
                                f"{given} {was_were} given"
                        )
                    }
                    result = base
                }
            }
        }
        TypeKind.Optional => { result = optional_of(c.universe, resolve_ast_type(c, t.inner)) }
        TypeKind.Union => {
            var members: list[Type?] = []
            for argument in t.args { members.push(resolve_ast_type(c, argument)) }
            result = if members.len() > 0 { union_of(c.universe, members) } else { c.universe.t_any }
        }
        TypeKind.Fn => {
            var params: list[Type] = []
            for argument in t.args { params.push(resolve_ast_type(c, argument) ?? c.universe.t_any) }
            result = fn_of(c.universe, params, resolve_ast_type(c, t.inner))
        }
        TypeKind.Tuple => {
            var members: list[Type] = []
            for argument in t.args { members.push(resolve_ast_type(c, argument) ?? c.universe.t_any) }
            result = tuple_of(c.universe, members)
        }
    }

    let answer = result ?? c.universe.t_any
    c.set_resolved_type(t, answer)
    return answer
}

#: Install what only this file can supply, then let `stmt` and `decl` install
#: theirs.
#:
#: Order matters only in that all three must run before anything is checked; a
#: `Ctx` with any callback still at its default answers `any` where it should
#: answer a type, and an `any` on both sides of an assignment needs no guard.
pub fn install(c: Ctx) -> void {
    c.resolve_ast_type = resolve_ast_type
    c.lookup_type_name = lookup_type_name
    stmt_mod.install(c)
    decl_mod.install(c)
}

#: Register every class, trait and enum this file declares.
#:
#: Before anything is checked, because a declaration may mention a sibling
#: declared later in the file and that sibling has to mean the sibling.
fn _register_all(c: Ctx, node: any) -> void {
    if node is Node {
        if node.kind == NodeKind.ClassDecl or
            node.kind == NodeKind.TraitDecl or
            node.kind == NodeKind.EnumDecl {
            let _ = register_decl(c, node)
        }
        for key in node.fields.keys() { _register_all(c, node.fields[key]) }
    } elif node is list {
        for item in node { _register_all(c, item) }
    }
}

#: Type-check a resolved program, and answer the context that holds the result.
#:
#: The passes are the C's, in the C's order. Registration first, so a name
#: declared anywhere in the file is visible everywhere in it. Then layout, in
#: full rather than on demand: half the questions the checker asks about a type
#: — is it iterable, does it implement that trait, is it callable — are asked
#: in code with no declaration in hand to lay out first, and a laid-out
#: declaration answers them all while an empty one silently answers "no". Then
#: the body.
#: `collect_dump` turns on the record of every decision, which is what
#: `--dump-sema` writes and `make sema-diff` compares against the C. It costs a
#: list append per typed node, so it is off unless asked for.
pub fn check_program(
    universe: Universe,
    program: Node?,
    resolution: Resolution,
    bag: DiagnosticBag,
    fold_constants: bool = true,
    collect_dump: bool = false
) -> Ctx {
    let c = Ctx(universe, bag, resolution)
    c.fold_constants = fold_constants
    if collect_dump { c.dump = [] }
    install(c)
    if program == null { return c }

    _register_all(c, program)
    for declared in c.all_decls() { layout_decl(c, declared) }
    c.check_block(c, program)
    # The C ends jaiCheckProgram with `program->type = gTypes.tVoid`, and void
    # is null, which is why a sema dump records the program itself as `null`.
    c.set_type(program, universe.t_void)
    if collect_dump { _dump_types(c, program) }
    return c
}

# Every node's final type, recorded after checking rather than as each type is
# assigned. Folding rewrites nodes and a cast replaces one in place, so a type
# recorded when it was assigned can be a type the finished tree does not hold.
# The C does the same walk, in `jaiSemaDumpTypes`.
fn _dump_types(c: Ctx, node: any) -> void {
    if node is Node {
        let ty = c.type_of(node)
        if ty != null and c.dump != null {
            c.dump.push(
                Record(node.span.start, node.span.end, kind_name(node.kind), "type", render(ty))
            )
        }
        for key in node.fields.keys() { _dump_types(c, node.fields[key]) }
    } elif node is list {
        for item in node { _dump_types(c, item) }
    }
}
