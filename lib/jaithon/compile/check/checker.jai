#: The checker: what turns a resolved tree into a typed one.
#:
#: The parts of `src/sema/check.c` the rest of this directory leaves to it:
#: annotation to semantic type, the type-name scope that resolves against, and
#: the program-level passes that run before anything is checked.
from jaithon.ast import Node, NodeKind, Span, TypeExpr, TypeKind, kind_name
from jaithon.compile.check.decl import layout_decl, push_generics, register_decl
from jaithon.compile.check.decl import signature_of
from jaithon.compile.check.dump import Record
from jaithon.compile.check.modsig import MODULE_EXT, ModuleSig, SigImport, SigLoader
from jaithon.compile.check.modsig import absolute, default_search_path, dirname
from jaithon.compile.check.ctx import Ctx
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

#: How deep the cycle walk follows imports before giving up. The C needs no
#: bound; this keeps a stack overflow here from reporting as a crash.
const MAX_IMPORT_DEPTH: int = 256

#: The names that are types without anything declaring them. `void` resolves to
#: `null` because the two are one type; bare `list` is `list[any]`, but a
#: tuple's arity is part of its type, so bare `tuple` is `any`.
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

#: Classes the runtime provides that no source file declares. Exactly
#: `kBuiltinClasses` in `src/sema/check.c`, in its order: a wrong entry answers
#: `any` here and a type there, an extra one is a name the C rejects.
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

#: The type a bare name means here, or null when nothing names one. Innermost
#: binding wins: a generic parameter shadows a class, which shadows an alias.
#: Inside a foreign scope a name means one of *that* module's declarations, and
#: one this file cannot see is not a mistake it may report.
pub fn lookup_type_name(c: Ctx, name: str?) -> Type? {
    if name == null { return null }

    let scoped = c.scoped_type_name(name)
    if scoped != null { return scoped }

    let scalar = _builtin_scalar(c.universe, name)
    if scalar != null { return scalar }

    if c.foreign_origin is not null {
        let foreign = _lookup_foreign_type_name(c, name)
        if foreign is not null { return foreign }
        return _builtin_class(c.universe, name) ?? c.universe.t_any
    }

    let declared = c.decl_named(name)
    if declared != null and declared.ty != null { return declared.ty }

    # An imported name the loader could not follow. Reporting E0402 here broke
    # every `from std.core import Iterable` — 80 of the 198 files
    # `--bootstrap-verify` walks.
    if name in c.imported_names { return c.universe.t_any }

    return _builtin_class(c.universe, name)
}

# `lookupForeignTypeName` in `src/sema/check.c`. What it finds is registered
# *invisible*: this file did not import it, so only the module that did may
# name it.
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
# Jaithon, not an identity one, and `ModuleSig` defines no `__eq__`.
fn _same_module(left: any?, right: any?) -> bool {
    if left is null or right is null { return left is null and right is null }
    return id(left) == id(right)
}

# `list`, `set`, `dict` and `tuple` written with arguments. False when it did
# not handle the name, so a user generic falls through to erasure.
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

fn _generic_arity(d: TypeDecl?) -> int {
    if d == null { return 0 }
    let node = d.node
    if node == null { return 0 }
    if "generics" not in node.fields { return 0 }
    let generics = node.fields["generics"]
    if generics is list { return generics.len() }
    return 0
}

#: The semantic type a written annotation means. Memoised per `TypeExpr` in
#: `Ctx`, which is the C's `AstType.resolved` field.
#:
#: Generics are erased: `Box[int]` checks its arity and then stands for the bare
#: `Box`, because the C never builds an instantiation and `make sema-diff` would
#: report the difference on every generic annotation.
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
                # Resolved for their own diagnostics; erasure throws them away.
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
#: theirs. All three must run before anything is checked: a callback still at
#: its default answers `any` where it should answer a type.
pub fn install(c: Ctx) -> void {
    c.resolve_ast_type = resolve_ast_type
    c.lookup_type_name = lookup_type_name
    c.is_module = _is_module
    c.module_member = _module_member
    stmt_mod.install(c)
    decl_mod.install(c)
    # After `decl.install`, because it wraps it: laying out a foreign
    # declaration resolves its annotations in its own module. `jaiChkLayoutDecl`
    # opens with the same `jaiChkForeignBegin`.
    c.layout_decl = _layout_in_origin
}

fn _layout_in_origin(c: Ctx, declared: TypeDecl?) -> void {
    if declared is null { return }
    let saved = c.enter_foreign(declared.origin)
    layout_decl(c, declared)
    c.leave_foreign(saved)
}

fn _is_module(c: Ctx, node: Node?) -> bool {
    if node is null or node.kind != NodeKind.Ident { return false }
    let ty = c.symbol_type(c.resolution.symbol_of(node))
    return ty is not null and ty.kind == TyKind.Module
}

#: What `m.name` denotes after `import m`: the type of the declaration the
#: signature loader holds, the same answer `from m import name` gives. Without
#: it every qualified call reads `any`.
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
    # Defined at run time, or private: `any` is the honest answer.
    if decl is null { return (u.t_any, null) }

    match decl.kind {
        NodeKind.FnDecl => { return (_foreign_function_type(c, found[1], decl), decl) }
        NodeKind.ClassDecl => { return (_module_type_member(c, decl), decl) }
        NodeKind.TraitDecl => { return (_module_type_member(c, decl), decl) }
        NodeKind.EnumDecl => { return (_module_type_member(c, decl), decl) }
        _ => { return (u.t_any, null) }
    }
}

# `m.Thing` after `import m`. Registration happened when the import was seen; a
# declaration reached any other way has none and is `any`.
fn _module_type_member(c: Ctx, decl: Node) -> Type {
    let registered = c.decl_for_node(decl)
    if registered is null { return c.universe.t_any }
    c.layout_decl(c, registered)
    return registered.ty ?? c.universe.t_any
}

# `jaiChkForeignFunctionType`: the `Fn` type of a declaration read out of
# `origin`, resolved there. Without the scope every foreign annotation is `any`.
fn _foreign_function_type(c: Ctx, origin: ModuleSig?, fn_node: Node) -> Type {
    let saved = c.enter_foreign(origin)
    let signature = signature_of(c, fn_node, false)
    c.leave_foreign(saved)
    return signature
}

#: Register every class, trait and enum this file declares, before anything is
#: checked: a declaration may mention a sibling declared later in the file.
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
#: The passes are the C's, in the C's order: registration, layout in full rather
#: than on demand, then the body. An unlaid declaration silently answers "no" to
#: half the questions the checker asks about a type.
pub fn check_program(
    universe: Universe,
    program: Node?,
    resolution: Resolution,
    bag: DiagnosticBag,
    fold_constants: bool = true,
    collect_dump: bool = false,
    path: str = "",
    # Names bound before anything in `program` is looked at. A file needs none:
    # it declares everything it can name. A prompt input needs the classes the
    # inputs before it declared, which are in no registry this compilation can
    # reach.
    outer_type_names: list[tuple[str, Type]] = [],
    # `--strict`: an unannotated parameter is an error (E0400).
    strict: bool = false
) -> Ctx {
    let c = Ctx(universe, bag, resolution)
    c.fold_constants = fold_constants
    c.strict = strict
    if collect_dump { c.dump = [] }
    install(c)
    for (name, ty) in outer_type_names { c.push_type_name(name, ty) }
    if program == null { return c }

    # This file's own declarations first, so they win over imported ones.
    _register_all(c, program)
    let loader = _loader_for(_search_roots(path))
    c.sig_loader = loader
    c.source_dir = _dir_of(path)
    _import_pass(c, program, loader, _dir_of(path))
    _alias_pass(c, program)
    # The loop re-reads the length because a layout can discover a type in a
    # third module and register it; `all_decls` hands back the live list.
    var at = 0
    while at < c.all_decls().len() {
        c.layout_decl(c, c.all_decls()[at])
        at += 1
    }
    _signature_pass(c, program)
    c.check_block(c, program)
    # The C ends `jaiCheckProgram` the same way, and void is null, which is why
    # a sema dump records the program itself as `null`.
    c.set_type(program, universe.t_void)
    if collect_dump { _dump_types(c, program) }
    return c
}

#: Signature loaders already built, keyed by the search path they serve.
#:
#: A loader per `compile_source` re-parsed every module's import closure once
#: per importing file: `--bootstrap-verify` 50s against 19s. Keyed rather than
#: shared because a loader hands out file ids and spans carry them. Signatures
#: only: a shared `Universe` would leak one file's interning into the next.
var _loaders: dict[str, SigLoader] = {}

#: Forget every cached loader, so the next compilation re-reads from disk.
pub fn clear_signature_cache() -> void {
    _loaders = {}
}

fn _loader_for(roots: list[str]) -> SigLoader {
    let key = "|".join(roots)
    if key in _loaders { return _loaders[key] }
    let built = SigLoader(roots)
    _loaders[key] = built
    return built
}

fn _dir_of(path: str) -> str {
    let cut = path.rfind("/")
    if cut < 0 { return "" }
    return path[:cut]
}

# Where to look for an imported module, given the file doing the importing.
# A dotted name resolves against a *root*, not the importing file's directory:
# `std.math` from `lib/std/random.jai` is `lib/std/math.jai`, found by searching
# `lib` and not `lib/std`. `__prim__.module_path()` hands over the directories
# this binary really resolves against. The file's own ancestors come first all
# the same, so a file under `tests/` can import a sibling by plain dotted name.
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
        # A binary without the primitive falls through to the reconstruction.
    }
    return default_search_path()
}

# Give every imported name the type its module declares. The C's `visitImports`
# and `bindImportedName`. Only the *signature* is read. A module that cannot be
# found is not an error here: the resolver already reported the bad import.
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

#: Report `E0801` when the imports reachable from `path` close a cycle.
#:
#: `checkImportCycles` in `src/runtime/module.c`, over the signature loader's
#: graph. Deliberately not part of `check_program`: a cycle is a property of a
#: program, not a translation unit, and folding it in made the self-hosted
#: compiler reject two files the C compiles.
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

# The entry file, parsed for its import list alone. No bag is passed: a syntax
# error in the file being scanned belongs to its own compilation.
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

# The imports the entry file writes; its own tree is already parsed.
fn _own_imports(program: Node) -> list[SigImport] {
    var edges: list[SigImport] = []
    for node in program.child_list("stmts") {
        if node.kind != NodeKind.Import and node.kind != NodeKind.FromImport { continue }
        let dotted = node.opt_text("path")
        if dotted is not null { edges.push(SigImport(dotted, node.span)) }
    }
    return edges
}

# Depth-first; false once a cycle is reported, since one E0801 says it all.
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

# "a -> b -> a", from the first appearance of the module imported again.
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

fn _stem_of(path: str) -> str {
    var base = path
    let cut = base.rfind("/")
    if cut >= 0 { base = base[cut + 1:] }
    if base.ends_with(MODULE_EXT) { base = base[:base.len() - MODULE_EXT.len()] }
    return if base == "" { path } else { base }
}

fn _tail_of(dotted: str) -> str {
    let cut = dotted.rfind(".")
    if cut < 0 { return dotted }
    return dotted[cut + 1:]
}

# `loadSigFor`: load a module's signature and register every type it declares.
# The whole module, because a name this file asked for can mention a sibling.
fn _load_sig_for(c: Ctx, loader: SigLoader, dotted: str, from_dir: str) -> ModuleSig? {
    let sig = loader.load(dotted, from_dir)
    if sig is null { return null }
    _register_module_types(c, sig)
    return sig
}

fn _register_module_types(c: Ctx, sig: ModuleSig) -> void {
    for entry in sig.types { let _ = _register_foreign(c, sig, entry.decl, null, false) }
}

# Register one class, trait or enum read out of `origin`. `registerForeign`.
# The type is interned under its QUALIFIED name — `std.str.StringBuilder`, not
# `StringBuilder` — so two modules that each declare a `Node` stay two types.
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

    # Already registered, perhaps invisibly by `_register_module_types`: reuse
    # that entry rather than make a second, so both spellings stay one type.
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
# `bindImportedName`. A name the module does not declare is left as `any`.
fn _bind_imported(c: Ctx, loader: SigLoader, sig: ModuleSig, item: dict[str, any]) -> void {
    let remote: any? = item.get("name", null)
    if remote is not str { return }
    let written: any? = item.get("alias", null)
    let alias: str? = if written is str { written } else { null }

    let found = loader.find(sig, remote)
    let decl = found[0]
    if decl is null { return }
    # Under a re-export, `find` answers the module that really declares it.
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

# Bind every `type X = ...` alias before anything resolves an annotation. The
# C's `visitAlias`. An alias is in neither the declaration registry nor the
# builtin tables, so without this `lookup_type_name` reports E0402 on every use
# — all 108 remaining disagreements on `lib/std/list.jai`.
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

# Give every function its signature before any body is checked. The C's
# `visitSignature`. Without it a function called above its own declaration
# answers `any`: a file is checked top to bottom and calls do not follow that.
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
# assigned: folding rewrites nodes and a cast replaces one in place. The C's
# `jaiSemaDumpTypes` does the same walk.
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
        # A call argument, a parameter, a match arm: these carry nodes without
        # being nodes, so a walk over Node and list alone skips them.
        for key in node.keys() { _dump_types(c, node[key]) }
    }
}
