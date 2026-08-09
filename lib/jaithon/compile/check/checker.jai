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
from jaithon.compile.check.decl import layout_decl, push_generics, register_decl
from jaithon.compile.check.decl import signature_of
from jaithon.compile.check.dump import Record
from jaithon.compile.check.modsig import ModuleSig, SigLoader, default_search_path
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
#:
#: Exactly `kBuiltinClasses` in `src/sema/check.c`, in its order. Not a list of
#: plausible exception names: a first draft here guessed twenty-five and was
#: wrong in both directions, carrying `Exception`, `NotImplementedError`,
#: `TimeoutError` and `UnicodeError` that the C does not know, and spelling the
#: divide-by-zero one `ZeroDivisionError` where the C says
#: `DivisionByZeroError`. Every wrong entry is a name that answers `any` here
#: and a type there, and every extra one is a name this front end accepts and
#: the C rejects.
pub let BUILTIN_CLASSES: list[str] = [
    "Error",
    "AssertionError",
    "ArithmeticError",
    "DivisionByZeroError",
    "OverflowError",
    "LookupError",
    "IndexError",
    "KeyError",
    "NameError",
    "TypeError",
    "ValueError",
    "ParseError",
    "AttributeError",
    "IOError",
    "FileNotFoundError",
    "PermissionError",
    "OSError",
    "RuntimeError",
    "RecursionError",
    "StopIteration",
    "ImportError",
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
    c.is_module = _is_module
    c.module_member = _module_member
    stmt_mod.install(c)
    decl_mod.install(c)
}

# True when `node` names an imported module rather than a value.
fn _is_module(c: Ctx, node: Node?) -> bool {
    if node is null or node.kind != NodeKind.Ident { return false }
    let ty = c.symbol_type(c.resolution.symbol_of(node))
    return ty is not null and ty.kind == TyKind.Module
}

#: What `m.name` denotes after `import m`.
#:
#: The import pass gave `m` a `Module` type carrying the dotted name, and the
#: signature loader still holds that module's declarations, so the member's
#: type is the declaration's type — the same answer `from m import name` would
#: have produced. Without this every qualified call reads `any`, which was the
#: rest of `num.jai` and `os.jai` once the plain imports were bound.
fn _module_member(c: Ctx, node: Node?, name: str) -> (Type, Node?) {
    let u = c.universe
    if node is null { return (u.t_any, null) }
    let module_type = c.symbol_type(c.resolution.symbol_of(node))
    if module_type is null or module_type.kind != TyKind.Module { return (u.t_any, null) }

    let dotted = module_type.name
    let loader = c.sig_loader
    if dotted is null or loader is null { return (u.t_any, null) }

    let sig = loader.load(dotted, c.source_dir)
    if sig is null { return (u.t_any, null) }

    let found = loader.find(sig, name)
    let decl = found[0]
    if decl is null { return (u.t_any, null) }

    let ty = _signature_type(c, decl)
    return (ty ?? u.t_any, decl)
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
    collect_dump: bool = false,
    path: str = ""
) -> Ctx {
    let c = Ctx(universe, bag, resolution)
    c.fold_constants = fold_constants
    if collect_dump { c.dump = [] }
    install(c)
    if program == null { return c }

    _register_all(c, program)
    # `path` is the file being checked, and an import resolves relative to the
    # directory holding it before it resolves against the search path.
    let loader = SigLoader(_search_roots(path, default_search_path()))
    c.sig_loader = loader
    c.source_dir = _dir_of(path)
    _import_pass(c, program, loader, _dir_of(path))
    _alias_pass(c, program)
    for declared in c.all_decls() { layout_decl(c, declared) }
    _signature_pass(c, program)
    c.check_block(c, program)
    # The C ends jaiCheckProgram with `program->type = gTypes.tVoid`, and void
    # is null, which is why a sema dump records the program itself as `null`.
    c.set_type(program, universe.t_void)
    if collect_dump { _dump_types(c, program) }
    return c
}

# The directory holding `path`, which is where a relative import resolves from.
fn _dir_of(path: str) -> str {
    let cut = path.rfind("/")
    if cut < 0 { return "" }
    return path[:cut]
}

# Where to look for an imported module, given the file doing the importing.
#
# A dotted name resolves against a *root*, not against the importing file's own
# directory: `std.math` imported from `lib/std/random.jai` is `lib/std/math.jai`,
# which is found by searching `lib` and not `lib/std`. The C gets its roots from
# the executable's location and `JAITHON_PATH`; neither is reachable from here
# — `os_argv()[0]` is the script, not the interpreter — so every ancestor of the
# importing file is offered instead, nearest first. That finds `lib` from
# `lib/std/random.jai` without knowing anything about the install layout, and a
# root that holds no matching module simply does not answer.
fn _search_roots(path: str, base: list[str]) -> list[str] {
    var roots: list[str] = []
    var dir = _dir_of(path)
    while dir != "" {
        roots.push(dir)
        dir = _dir_of(dir)
    }
    for entry in base { roots.push(entry) }
    return roots
}

# Give every imported name the type the module it came from declares.
#
# The C's `visitImports` and `bindImportedName`. Without it every name another
# module exports is `any` here, which erases the type of every call through it
# and, in a standard library where files import each other constantly, is by
# far the largest single source of disagreement: measured against the C, it was
# 173 of `num.jai`'s 192 remaining, 163 of `json.jai`'s 179 and 140 of
# `test.jai`'s 169.
#
# Only the *signature* is read — `modsig` parses the other file and takes its
# declarations without checking its body, exactly as the C does. A module that
# cannot be found is not an error here: the resolver already reported an
# unresolvable import, and a second diagnostic about the same line helps
# nobody.
fn _import_pass(c: Ctx, node: any, loader: SigLoader, from_dir: str) -> void {
    if node is Node {
        if node.kind == NodeKind.Import {
            let dotted = node.opt_text("path")
            if dotted is not null {
                let sig = loader.load(dotted, from_dir)
                if sig is not null {
                    let symbol = c.resolution.symbol_of(node)
                    if symbol is not null and c.symbol_type(symbol) is null {
                        c.set_symbol_type(symbol, named(c.universe, TyKind.Module, dotted))
                    }
                }
            }
        } elif node.kind == NodeKind.FromImport {
            let dotted = node.opt_text("path")
            if dotted is not null {
                let sig = loader.load(dotted, from_dir)
                if sig is not null {
                    for item in node.records("items") { _bind_imported(c, loader, sig, item) }
                }
            }
        }
        for key in node.fields.keys() { _import_pass(c, node.fields[key], loader, from_dir) }
    } elif node is list {
        for item in node { _import_pass(c, item, loader, from_dir) }
    } elif node is dict {
        for key in node.keys() { _import_pass(c, node[key], loader, from_dir) }
    }
}

# One `name` or `name as local` out of a `from ... import` list.
fn _bind_imported(c: Ctx, loader: SigLoader, sig: ModuleSig, item: dict[str, any]) -> void {
    let remote = if "name" in item { item["name"] } else { null }
    if remote is not str { return }

    let found = loader.find(sig, remote)
    let decl = found[0]
    if decl is null { return }

    # The declaring module's own name scope, not this one's: a signature is
    # resolved where it was written, so a type it mentions means what it meant
    # there. Anything this file cannot see falls back to `any`, which is what
    # the C does for a foreign name too.
    let ty = _signature_type(c, decl)
    if ty is null { return }

    let symbol = c.resolution.symbol_of(item)
    if symbol is not null and c.symbol_type(symbol) is null {
        c.set_symbol_type(symbol, ty)
    }
}

# The type a declaration read out of another module's signature denotes.
fn _signature_type(c: Ctx, decl: Node) -> Type? {
    match decl.kind {
        NodeKind.FnDecl => { return signature_of(c, decl, false) }
        NodeKind.ClassDecl => { return named(c.universe, TyKind.Class, decl.opt_text("name")) }
        NodeKind.TraitDecl => { return named(c.universe, TyKind.Trait, decl.opt_text("name")) }
        NodeKind.EnumDecl => { return named(c.universe, TyKind.Enum, decl.opt_text("name")) }
        NodeKind.TypeDecl => { return resolve_ast_type(c, decl.type_field("aliased")) }
        _ => { return null }
    }
}

# Bind every `type X = ...` alias before anything resolves an annotation.
#
# The C's `visitAlias`. An alias is a name that stands for a type without
# declaring one, so it is in neither the declaration registry nor the builtin
# tables, and `lookup_type_name` would answer null and report E0402 on every
# use. Measured on `lib/std/list.jai`, that was all 108 remaining
# disagreements: `Less[T]` and `Rng` read as `any`, which erased the signature
# of every function taking one, which erased the type of every call through it.
#
# A generic alias binds its own parameters while its target is resolved, so
# `type Less[T] = fn(T, T) -> bool` resolves to `fn(T, T) -> bool` with `T` a
# generic parameter, and erasure at the use site then does the rest.
fn _alias_pass(c: Ctx, node: any) -> void {
    if node is Node {
        if node.kind == NodeKind.TypeDecl {
            let name = node.opt_text("name")
            if name is not null {
                let mark = c.type_name_mark()
                push_generics(c, node.records("generics"))
                let target = resolve_ast_type(c, node.type_field("aliased"))
                c.type_name_restore(mark)
                if target is not null {
                    c.push_type_name(name, target)
                    c.set_symbol_type(c.resolution.symbol_of(node), target)
                }
            }
        }
        for key in node.fields.keys() { _alias_pass(c, node.fields[key]) }
    } elif node is list {
        for item in node { _alias_pass(c, item) }
    } elif node is dict {
        for key in node.keys() { _alias_pass(c, node[key]) }
    }
}

# Give every function its signature before any body is checked.
#
# The C's `visitSignature`. Without it a function called above its own
# declaration answers `any`, because the symbol only acquires a type when
# `check_decl` reaches the declaration — and a file is checked top to bottom
# while calls do not respect that order. Measured against the C on
# `lib/std/math.jai`, this was every remaining disagreement: `fn(int) -> int`
# and `fn(list[float]) -> float` read as `any` at the call site, which then
# lost the return type, which then lost the guard on the value it returned.
fn _signature_pass(c: Ctx, node: any) -> void {
    if node is Node {
        if node.kind == NodeKind.FnDecl {
            let symbol = c.resolution.symbol_of(node)
            if symbol is not null and c.symbol_type(symbol) is null {
                c.set_symbol_type(symbol, signature_of(c, node, false))
            }
        }
        for key in node.fields.keys() { _signature_pass(c, node.fields[key]) }
    } elif node is list {
        for item in node { _signature_pass(c, item) }
    } elif node is dict {
        for key in node.keys() { _signature_pass(c, node[key]) }
    }
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
    } elif node is dict {
        # A sub-structure record — a call argument, a parameter, a match arm.
        # These carry nodes without being nodes, so a walk that follows only
        # Node and list skips every expression passed as an argument. That is
        # not a small corner: it hid five of the basic fixture's decisions, and
        # every one of them was a decision the checker had already made
        # correctly.
        for key in node.keys() { _dump_types(c, node[key]) }
    }
}
