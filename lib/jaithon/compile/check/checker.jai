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
from jaithon.ast import Node, NodeKind, Span, TypeExpr, TypeKind, kind_name
from jaithon.compile.check.decl import layout_decl, push_generics, register_decl
from jaithon.compile.check.decl import signature_of
from jaithon.compile.check.dump import Record
from jaithon.compile.check.modsig import MODULE_EXT, ModuleSig, SigImport, SigLoader
from jaithon.compile.check.modsig import absolute, default_search_path, dirname
from jaithon.compile.check.expr import Ctx
from jaithon.compile.check.render import render
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.union import optional_of, union_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, list_of
from jaithon.compile.check.universe import named, set_of, tuple_of
from jaithon.compile.lexer import DiagnosticBag
from jaithon.compile.parser import parse
from jaithon.compile.resolve import Resolution
import jaithon.compile.check.decl as decl_mod
import jaithon.compile.check.stmt as stmt_mod

#: An annotation named a type nothing declares.
pub const E0402_UNKNOWN_TYPE: str = "E0402"

#: A generic was written with the wrong number of arguments.
pub const E0407_GENERIC_ARITY: str = "E0407"

#: The imports reachable from this file close a loop.
pub const E0801_CIRCULAR_IMPORT: str = "E0801"

#: How deep the cycle walk follows imports before giving up.
#:
#: The C bounds nothing here — its graph is a real graph with a colour per
#: node, so the walk terminates on its own. This one bounds the *recursion*,
#: because a Jaithon stack overflow in the checker would report as a crash
#: rather than as the answer "no cycle found within reach".
const MAX_IMPORT_DEPTH: int = 256

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
#:
#: Inside a foreign scope the whole tail of the search changes, because the
#: question changes: a name written in another module's source means one of
#: *that* module's declarations, never one of this file's, and a name this file
#: cannot see is not a mistake it may report. `jaiChkLookupTypeName` splits the
#: same way.
pub fn lookup_type_name(c: Ctx, name: str?) -> Type? {
    if name == null { return null }

    let scoped = c.scoped_type_name(name)
    if scoped != null { return scoped }

    let scalar = _builtin_scalar(c.universe, name)
    if scalar != null { return scalar }

    if c.foreign_origin is not null {
        let foreign = _lookup_foreign_type_name(c, name)
        if foreign is not null { return foreign }
        # Never null: an unresolvable name in a foreign signature is a name
        # this file cannot see, not a mistake it may report.
        return _builtin_class(c.universe, name) ?? c.universe.t_any
    }

    let declared = c.decl_named(name)
    if declared != null and declared.ty != null { return declared.ty }

    # A name this file imported that names no declaration the loader could
    # follow: one the module defines at run time, or one whose module is not on
    # the search path. It is a real name, not a typo, and gradual typing
    # already has a word for a type whose shape is unknown. Reporting E0402
    # here would make every `from std.core import Iterable` unusable in an
    # annotation — which is exactly what it did, on 80 of the 198 files
    # `--bootstrap-verify` walks.
    if name in c.imported_names { return c.universe.t_any }

    return _builtin_class(c.universe, name)
}

# A name written in another module's source. It means what it means *there*:
# one of that module's own declarations, or one it imported in turn — followed
# through, because otherwise `fn draw(w: Window)` in `std.gui.canvas` would
# check nothing and the difference between a local call and a call one module
# away would be exactly the thing this is here to remove.
#
# `lookupForeignTypeName` in `src/sema/check.c`. What it finds is registered
# *invisible*: this file did not import it, so it may not be named here — it is
# registered only so that the module that did import it can.
fn _lookup_foreign_type_name(c: Ctx, name: str) -> Type? {
    let origin = c.foreign_origin
    for entry in c.all_decls() {
        if not _same_module(entry.origin, origin) { continue }
        if entry.name == name { return entry.ty }
    }

    let loader = c.sig_loader
    if loader is null or origin is null { return null }
    let found = loader.find(origin, name)
    let decl = found[0]
    if decl is null { return null }
    let registered = _register_foreign(c, found[1], decl, null, false)
    if registered is null { return null }
    return registered.ty
}

# Whether two `ModuleSig?` values are the same module. `is` is a type test in
# Jaithon, not an identity one, and `ModuleSig` defines no `__eq__`, so `==`
# would be identity — but only for two non-null operands.
fn _same_module(left: any?, right: any?) -> bool {
    if left is null or right is null { return left is null and right is null }
    return id(left) == id(right)
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
    # After `decl.install`, because it wraps what that installed: laying a
    # declaration out resolves every annotation in its body, and one read out
    # of another module has to resolve them there. `jaiChkLayoutDecl` opens
    # with the same `jaiChkForeignBegin(c, entry->decl->origin, ...)`.
    c.layout_decl = _layout_in_origin
}

# `decl.layout_decl`, with the declaration's own module installed around it.
fn _layout_in_origin(c: Ctx, declared: TypeDecl?) -> void {
    if declared is null { return }
    let saved = c.enter_foreign(declared.origin)
    layout_decl(c, declared)
    c.leave_foreign(saved)
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
    # Defined at run time, or private: `any` is the honest answer and the one
    # every imported name used to get.
    if decl is null { return (u.t_any, null) }

    match decl.kind {
        NodeKind.FnDecl => { return (_foreign_function_type(c, found[1], decl), decl) }
        NodeKind.ClassDecl => { return (_module_type_member(c, decl), decl) }
        NodeKind.TraitDecl => { return (_module_type_member(c, decl), decl) }
        NodeKind.EnumDecl => { return (_module_type_member(c, decl), decl) }
        _ => { return (u.t_any, null) }
    }
}

# `m.Thing` after `import m`: the registered foreign declaration, laid out, so
# that `m.Thing(...)` constructs and `m.Thing.member` resolves. Registration
# happened when the import was seen; a declaration reached any other way has
# none and is `any`.
fn _module_type_member(c: Ctx, decl: Node) -> Type {
    let registered = c.decl_for_node(decl)
    if registered is null { return c.universe.t_any }
    c.layout_decl(c, registered)
    return registered.ty ?? c.universe.t_any
}

# The `Fn` type of a declaration read out of `origin`, resolved there.
#
# `jaiChkForeignFunctionType`. Without the scope, `fn verify_source(...) ->
# VerifyResult` resolved in this file's scope gives `fn(...) -> any`, and the
# `any` then spreads to every call, every binding that holds a result, and
# every member read off one — 75 of the 137 decisions in
# `tests/lang/test_bootstrap_verify.jai` alone.
fn _foreign_function_type(c: Ctx, origin: ModuleSig?, fn_node: Node) -> Type {
    let saved = c.enter_foreign(origin)
    let signature = signature_of(c, fn_node, false)
    c.leave_foreign(saved)
    return signature
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

    # This file's own declarations first, so that a name it declares always
    # wins over one it imports, and only then what the imports bring in.
    _register_all(c, program)
    # `path` is the file being checked, and an import resolves relative to the
    # directory holding it before it resolves against the search path.
    let loader = _loader_for(_search_roots(path))
    c.sig_loader = loader
    c.source_dir = _dir_of(path)
    _import_pass(c, program, loader, _dir_of(path))
    _alias_pass(c, program)
    # Everything registered, imported declarations included. Laying those out
    # on demand does not work: `Iter` is assignable to `Iterator` only if its
    # trait list says so, and half the questions the checker asks about a type
    # are asked in code with no declaration in hand to lay out first.
    #
    # The loop re-reads the length because a layout can discover a type in a
    # third module and register it, which is why `all_decls` hands back the
    # live list rather than a copy.
    var at = 0
    while at < c.all_decls().len() {
        c.layout_decl(c, c.all_decls()[at])
        at += 1
    }
    _signature_pass(c, program)
    c.check_block(c, program)
    # The C ends jaiCheckProgram with `program->type = gTypes.tVoid`, and void
    # is null, which is why a sema dump records the program itself as `null`.
    c.set_type(program, universe.t_void)
    if collect_dump { _dump_types(c, program) }
    return c
}

#: Signature loaders already built, keyed by the search path they serve.
#:
#: A signature is a pure function of the file it came from, so parsing one
#: twice is waste. Every `compile_source` built its own loader, so a run over
#: the whole tree re-parsed each module's entire import closure once per
#: importing file: `--bootstrap-verify` went 25s -> 50s the moment the checker
#: started following imports, and back to 19s with this. The C keeps the same
#: table process-global (`sSigs`, src/sema/modsig.c).
#:
#: Keyed by search path rather than shared outright, because a loader hands out
#: file ids and spans carry them — two differently configured loaders must not
#: trade signatures and leave a span pointing at the wrong file.
#:
#: This caches signatures. It never caches types: a shared `Universe` would make
#: one file's compilation depend on what an earlier one interned, which is the
#: cross-file contamination the C has and this front end does not.
var _loaders: dict[str, SigLoader] = {}

#: Forget every cached loader, for a caller that changed a file on disk and
#: wants the next compilation to see it.
pub fn clear_signature_cache() -> void {
    _loaders = {}
}

# The loader for this search path, built once.
fn _loader_for(roots: list[str]) -> SigLoader {
    let key = "|".join(roots)
    if key in _loaders { return _loaders[key] }
    let built = SigLoader(roots)
    _loaders[key] = built
    return built
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
# which is found by searching `lib` and not `lib/std`.
#
# `__prim__.module_path()` is the answer to that, and it is the whole reason the
# primitive exists: it hands over the directories this binary actually resolves
# an import against — `JAITHON_PATH`, the ones derived from the executable's
# own location, the fixed system ones and anything `-I` added — so the two
# front ends agree about where a module lives instead of one of them guessing
# from the source path. `default_search_path` reconstructs a subset of that
# list and is the fallback for a build where the primitive is missing.
#
# The importing file's own ancestors come first all the same. That is not the
# runtime's rule and it is not there to find `std`: it is there so that a file
# under `tests/` can import a sibling package by a plain dotted name, which the
# fixtures do and which the C's `-I` handles for it.
fn _search_roots(path: str) -> list[str] {
    var roots: list[str] = []
    var dir = _dir_of(path)
    while dir != "" {
        roots.push(dir)
        dir = _dir_of(dir)
    }
    for entry in _runtime_module_path() { roots.push(entry) }
    return roots
}

# The directories this binary resolves imports against, or as much of them as
# can be reconstructed when the primitive is not there.
fn _runtime_module_path() -> list[str] {
    try {
        let given = __prim__.module_path()
        if given is list {
            var out: list[str] = []
            for entry in given {
                if entry is str { out.push(entry) }
            }
            if out.len() > 0 { return out }
        }
    } catch _missing: Error {
        # A binary without the primitive: reconstruct what can be seen.
    }
    return default_search_path()
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
                c.imported_names.add(node.opt_text("alias") ?? _tail_of(dotted))
                let sig = _load_sig_for(c, loader, dotted, from_dir)
                if sig is not null {
                    let symbol = c.resolution.symbol_of(node)
                    if symbol is not null and c.symbol_type(symbol) is null {
                        c.set_symbol_type(symbol, named(c.universe, TyKind.Module, dotted))
                    }
                }
            }
        } elif node.kind == NodeKind.FromImport {
            # `from m import *` binds no symbols here at all — the resolver
            # defers every unknown name to the run time — so there is nothing
            # to attach.
            if not node.flag("isWildcard") {
                for item in node.records("items") {
                    let remote: any? = item.get("name", null)
                    if remote is not str { continue }
                    let alias: any? = item.get("alias", null)
                    c.imported_names.add(if alias is str { alias } else { remote })
                }
                let dotted = node.opt_text("path")
                if dotted is not null {
                    let sig = _load_sig_for(c, loader, dotted, from_dir)
                    if sig is not null {
                        for item in node.records("items") { _bind_imported(c, loader, sig, item) }
                    }
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

# ------------------------------------------------------------------
# Import cycles
# ------------------------------------------------------------------

#: Report `E0801` into `bag` when the imports reachable from `path` close a
#: cycle.
#:
#: `checkImportCycles` in `src/runtime/module.c`, over the signature loader's
#: graph instead of a second one built for the purpose. The two are the same
#: graph: a `ModuleSig` is a parsed file, and `ModuleSig.imports` is exactly
#: the edge list `collectImports` builds.
#:
#: **Deliberately not part of `check_program`.** A cycle is a property of a
#: *program*, not of a translation unit, and the C draws the line in the same
#: place: `jaiCheckFile` runs this before the front end and `jaiCompileSource`
#: never runs it at all. Folding it into the front end would make the
#: self-hosted compiler reject two files the C compiles happily — which is
#: exactly what `--bootstrap-verify` then reported, and correctly.
#:
#: Nothing about a module two steps away is reported. A module that cannot be
#: found contributes no edge — E0800 is the importer's diagnostic, at import
#: time — and a module that does not parse contributes an empty one, because a
#: file with a syntax error has no reliable import list and will report its own
#: E01xx when something actually compiles it.
pub fn check_import_cycles(bag: DiagnosticBag, path: str, program: Node? = null) -> void {
    if path == "" { return }
    let tree = program ?? _parse_quietly(path)
    if tree is null { return }

    let c = Ctx(Universe(), bag)
    let loader = _loader_for(_search_roots(path))
    var visited: set[str] = set()
    var stack: list[str] = [absolute(path)]
    var names: list[str] = [_stem_of(path)]
    var entered: list[Span] = [Span.none()]
    let _complete = _walk_imports(
        c,
        loader,
        _own_imports(tree),
        _dir_of(path),
        stack,
        names,
        entered,
        visited
    )
}

# The entry file, parsed for its import list alone. No bag is passed, so
# whatever the parse has to say dies with the call: a syntax error in the file
# being scanned belongs to its own compilation.
fn _parse_quietly(path: str) -> Node? {
    let text = _read_text(path)
    if text is null { return null }
    let tree = parse(text, 0)
    if tree is null or tree.kind != NodeKind.Program { return null }
    return tree
}

fn _read_text(path: str) -> str? {
    try {
        let handle = __prim__.io_open(path, "rb")
        let raw = __prim__.io_read(handle, -1)
        __prim__.io_close(handle)
        return __prim__.str_decode(raw)
    } catch _failure: Error {
        return null
    }
}

# The imports the entry file writes. Its own tree is already parsed, so there
# is nothing to load: only a module reached *through* an import needs a
# signature.
fn _own_imports(program: Node) -> list[SigImport] {
    var edges: list[SigImport] = []
    for node in program.child_list("stmts") {
        if node.kind != NodeKind.Import and node.kind != NodeKind.FromImport { continue }
        let dotted = node.opt_text("path")
        if dotted is not null { edges.push(SigImport(dotted, node.span)) }
    }
    return edges
}

# Depth-first over `edges`, answering false once a cycle has been reported: one
# E0801 says everything the rest of the walk would repeat.
fn _walk_imports(
    c: Ctx,
    loader: SigLoader,
    edges: list[SigImport],
    from_dir: str,
    stack: list[str],
    names: list[str],
    entered: list[Span],
    visited: set[str]
) -> bool {
    if stack.len() >= MAX_IMPORT_DEPTH { return true }
    for edge in edges {
        let sig = loader.load(edge.dotted, from_dir)
        if sig is null { continue }
        if sig.path in visited { continue }

        var at = -1
        for (index, entry) in stack.enumerate() {
            if entry == sig.path {
                at = index
                break
            }
        }
        if at >= 0 {
            _report_cycle(c, names, entered, at, edge)
            return false
        }

        stack.push(sig.path)
        names.push(_stem_of(sig.path))
        entered.push(edge.span)
        let complete = _walk_imports(
            c,
            loader,
            sig.imports,
            dirname(sig.path),
            stack,
            names,
            entered,
            visited
        )
        let _left = stack.pop()
        let _unnamed = names.pop()
        let _edge = entered.pop()
        if not complete { return false }
        visited.add(sig.path)
    }
    return true
}

# "a -> b -> a", starting at the first appearance of the module that is being
# imported again, so the cycle itself is what the reader gets.
fn _report_cycle(
    c: Ctx,
    names: list[str],
    entered: list[Span],
    at: int,
    closing: SigImport
) -> void {
    var chain: list[str] = []
    var index = at
    while index < names.len() {
        chain.push(names[index])
        index += 1
    }
    chain.push(names[at])

    let d = c.error(E0801_CIRCULAR_IMPORT, closing.span, f"circular import: {" -> ".join(chain)}")
    d.label(closing.span, "this import closes the cycle")
    # The edge that entered the cycle, so both ends of it are on screen.
    if at + 1 < entered.len() { d.label(entered[at + 1], "the cycle starts here") }
    d.with_help(
        "move the shared declarations into a module every step of the cycle " +
            "can import, or move the import inside the function that needs it"
    )
}

# The name a file answers to in a cycle report: its basename without `.jai`.
fn _stem_of(path: str) -> str {
    var base = path
    let cut = base.rfind("/")
    if cut >= 0 { base = base[cut + 1:] }
    if base.ends_with(MODULE_EXT) { base = base[:base.len() - MODULE_EXT.len()] }
    return if base == "" { path } else { base }
}

# The last component of a dotted module name: what `import std.gui` binds.
fn _tail_of(dotted: str) -> str {
    let cut = dotted.rfind(".")
    if cut < 0 { return dotted }
    return dotted[cut + 1:]
}

# Load a module's signature and register every type it declares.
#
# `loadSigFor`. Registration covers the whole module and not only the names
# this file asked for, because a declaration it *did* ask for can mention a
# sibling in an annotation and that sibling has to mean the sibling.
fn _load_sig_for(c: Ctx, loader: SigLoader, dotted: str, from_dir: str) -> ModuleSig? {
    let sig = loader.load(dotted, from_dir)
    if sig is null { return null }
    _register_module_types(c, sig)
    return sig
}

fn _register_module_types(c: Ctx, sig: ModuleSig) -> void {
    for entry in sig.types { let _ = _register_foreign(c, sig, entry.decl, null, false) }
}

# Register one class, trait or enum read out of `origin`.
#
# `registerForeign`. The type is interned under its QUALIFIED name —
# `std.str.StringBuilder`, not `StringBuilder` — which is what keeps two
# modules that each declare a `Node` from being one type, and a `Window`
# declared here from being `std.gui.Window`. Interning imported classes under
# their bare name was every disagreement in `json.jai` and `os.jai` once
# imports were bound: the C says `std.io.File` and this said `File`, which is a
# different type with the same spelling.
fn _register_foreign(
    c: Ctx,
    origin: ModuleSig?,
    node: Node,
    alias: str?,
    visible: bool
) -> TypeDecl? {
    var kind = TyKind.Class
    match node.kind {
        NodeKind.ClassDecl => { kind = TyKind.Class }
        NodeKind.TraitDecl => { kind = TyKind.Trait }
        NodeKind.EnumDecl => { kind = TyKind.Enum }
        _ => { return null }
    }

    let name = node.opt_text("name")
    if name is null { return null }
    let binding = alias ?? name

    # Already registered, perhaps invisibly by `_register_module_types`: an
    # import that names it makes the same entry reachable rather than making a
    # second one, so the two spellings stay one type.
    let existing = c.decl_for_node(node)
    if existing is not null {
        if visible { c.register_decl(existing, true, binding) }
        return existing
    }

    let declared = TypeDecl(name, node.span, node)
    declared.origin = origin
    declared.is_trait = node.kind == NodeKind.TraitDecl
    declared.is_enum = node.kind == NodeKind.EnumDecl
    declared.is_abstract = node.kind == NodeKind.ClassDecl and node.flag("isAbstract")
    let qualified = if origin is null { name } else { origin.qualify(name) }
    declared.ty = named(c.universe, kind, qualified, declared)
    c.register_decl(declared, visible, binding)
    return declared
}

# One `name` or `name as local` out of a `from ... import` list.
#
# `bindImportedName`. A name the module does not declare — one its body defines
# at run time, or one this loader cannot follow — is left exactly as it was:
# `any`, which is what every imported name used to be.
fn _bind_imported(c: Ctx, loader: SigLoader, sig: ModuleSig, item: dict[str, any]) -> void {
    let remote: any? = item.get("name", null)
    if remote is not str { return }
    let written: any? = item.get("alias", null)
    let alias: str? = if written is str { written } else { null }

    let found = loader.find(sig, remote)
    let decl = found[0]
    if decl is null { return }
    # `find` answers the module that really declares the name, which under a
    # re-export is not the one that was imported from.
    let owner = found[1]
    if not _same_module(owner, sig) { _register_module_types(c, owner) }

    let symbol = c.resolution.symbol_of(item)
    match decl.kind {
        NodeKind.FnDecl => {
            c.set_imported_decl(symbol, decl)
            if c.symbol_type(symbol) is null {
                c.set_symbol_type(symbol, _foreign_function_type(c, owner, decl))
            }
        }
        NodeKind.TypeDecl => {
            let saved = c.enter_foreign(owner)
            let mark = c.type_name_mark()
            push_generics(c, decl.records("generics"))
            let target = resolve_ast_type(c, decl.type_field("aliased")) ?? c.universe.t_any
            c.type_name_restore(mark)
            c.leave_foreign(saved)
            # Pushed before this file's own aliases, so a local one still wins.
            c.push_type_name(alias ?? remote, target)
            if c.symbol_type(symbol) is null { c.set_symbol_type(symbol, target) }
        }
        _ => {
            let registered = _register_foreign(c, owner, decl, alias, true)
            if registered is null { return }
            c.set_imported_decl(symbol, decl)
            if c.symbol_type(symbol) is null { c.set_symbol_type(symbol, registered.ty) }
        }
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
