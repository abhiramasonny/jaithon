#: The self-hosted compiler front end: source in, `.jaic` image out.
#:
#: `compile_source` runs the whole pipeline — lexer, parser, resolver, emitter,
#: container writer — through one `DiagnosticBag`, so a file's diagnostics come
#: out in source order whichever stage raised them, and hands back everything
#: each stage produced instead of printing anything itself.
#:
#: `compile_repl` is the same pipeline with the three differences a prompt
#: needs: one line read two ways, a resolver that defers unknown globals, and a
#: `ReplSession` carrying the type names earlier inputs declared.
#:
#: This file used to hold the other half of a differential oracle —
#: `compare_functions`, `verify`, `verify_source` — which compiled a file with
#: both front ends and compared the records field by field. It was retired with
#: the rest of that mechanism once there was only one front end left to compare.
#:
#: ```
#: let built = compile_source("let answer = 42\n", "<doc>")
#: built.ok()                       # true
#: built.image.len() > 32           # true
#: ```
#:
#: The file system is reached through `__prim__.io_open/read/write/flush/close`,
#: and byte strings through `__prim__.bytes_new`.
from jaithon.ast import Node, NodeKind
from jaithon.compile.emit import ConstTag, FuncProto, emit, op_name
from jaithon.compile.jaic import FLAG_DEBUG, FLAG_RELEASE, Module, disassemble
from jaithon.compile.jaic import disassemble_instruction
from jaithon.compile.jaic import source_hash, write_module
from jaithon.compile.check.checker import check_import_cycles, check_program
from jaithon.compile.check.ctx import Ctx
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.ty import Type
from jaithon.compile.check.universe import Universe, named
from jaithon.compile.lexer import Diagnostic, DiagnosticBag, Lexer, Severity
from jaithon.compile.opt import optimize
from jaithon.compile.parser import Parser, parse
from jaithon.compile.resolve import Resolution, SemaCode, resolve

#: Everything one compilation produced.
#:
#: `image` is empty when a stage failed and `diagnostics` then says why. The
#: body is still returned when there is one, because a half-compiled program is
#: worth disassembling while its errors are being read.
pub class Compiled {
    #: The path the source was read from, or a marker like `"<doc>"`.
    pub let path: str
    #: FNV-1a 64 of the source text, which keys the cache.
    pub let hash: int
    #: Errors and warnings from every stage, in source order.
    pub let diagnostics: DiagnosticBag
    #: The parsed tree, or null when parsing failed.
    pub var program: Node?
    #: The resolver's symbol table, or null.
    pub var resolution: Resolution?
    #: The module body's function record, or null when nothing was emitted.
    pub var body: FuncProto?
    #: What the checker decided, or null when it did not run.
    pub var types: Ctx?
    #: The container description, or null.
    pub var mod: Module?
    #: The serialised `.jaic` bytes; empty on failure.
    pub var image: list[int]
    #: True when a prompt input read as a single expression, which is what
    #: `:type` requires and `compile_source` never asks about.
    pub var repl_expression: bool

    #: Create an empty result for `path`.
    pub fn init(self, path: str, hash: int, diagnostics: DiagnosticBag) {
        self.path = path
        self.hash = hash
        self.diagnostics = diagnostics
        self.program = null
        self.resolution = null
        self.body = null
        self.types = null
        self.mod = null
        self.image = []
        self.repl_expression = false
    }

    #: True when no stage reported an error.
    pub fn ok(self) -> bool { return not self.diagnostics.has_errors() }

    #: The diagnostics rendered one per line, ready to print.
    pub fn report(self) -> str {
        var lines: list[str] = []
        for diagnostic in self.diagnostics.items {
            lines.push(f"{self.path}: {diagnostic}")
        }
        return "\n".join(lines)
    }
}

#: Compile `text` as the module at `path`.
#:
#: `file_id` is the index `path` has in the caller's source table; spans carry
#: it so that a diagnostic can be pointed back at the right file. Nothing is
#: written to disk.
#:
#: `opt_level` is the `--O` level the emitted chunks are optimised at, and 2 is
#: what the C driver defaults `CodegenOptions.optLevel` to. The C bridge in
#: `src/runtime/module.c` calls this with `(source, path)` and nothing else, so
#: the default is the level `--front=jai` and `make bootstrap` both run at.
#:
#: ```
#: compile_source("let x = 1\n", "<doc>").ok()          # true
#: compile_source("let x = y\n", "<doc>").ok()          # false, E0200
#: ```
pub fn compile_source(
    text: str,
    path: str,
    release: bool = false,
    file_id: int = 0,
    opt_level: int = 2
) -> Compiled {
    let bag = DiagnosticBag()
    let result = Compiled(path, source_hash(text), bag)

    let program = parse(text, file_id, bag)
    if program == null { return result }
    result.program = program

    let resolution = resolve(program, bag)
    result.resolution = resolution

    # The checker rewrites the tree it is handed: it folds constant expressions
    # in place and wraps a value in a `Cast` wherever a runtime guard or an
    # `int -> float` widening is needed. The emitter turns those into
    # `OP_TYPE_GUARD` and `OP_TO_FLOAT`, which is where about half the bytecode
    # the two front ends disagree about comes from.
    let checked = check_program(Universe(), program, resolution, bag, opt_level > 0, path)
    result.types = checked

    # The emitter is handed the optimiser rather than importing it: the passes
    # read `jaithon.compile.emit`, so the module that runs them cannot be
    # imported from there. See `_no_optimize` in that file.
    let body = emit(
        program,
        resolution,
        bag,
        release,
        module_name_for(path),
        opt_level,
        optimize,
        text,
        checked.enum_decls()
    )
    result.body = body
    if not result.ok() { return result }

    let flags = if release { FLAG_RELEASE } else { FLAG_DEBUG }
    let mod = Module(path, result.hash, body, flags)
    for name in _exported_names(body) { mod.exports.push(name) }
    result.mod = mod
    result.image = write_module(mod)
    return result
}

#: A prompt session: what one input leaves behind for the next.
#:
#: A file is compiled once and sees every declaration it can name. A prompt is
#: compiled a line at a time, and line two has to be able to name the class
#: line one declared -- in an annotation, where the resolver's deferred-global
#: rule does not reach, because that rule is about values and an annotation
#: asks about types. The C driver kept this list in the C resolver's built-in
#: registry; with that registry gone, the session keeps it.
pub class ReplSession {
    #: What each name this session declared was declared as.
    pub var types: dict[str, TyKind]

    pub fn init(self) { self.types = {} }

    #: Forget every declaration, for `:reset`.
    pub fn reset(self) -> void { self.types = {} }

    #: Record the type names one input introduces.
    #:
    #: Read off the checker rather than the tree, because `from m import Point`
    #: binds a name to a class as surely as `class Point` does and the tree
    #: alone cannot say which -- only the checker has been to `m`. The C driver
    #: had a second walk (`registerImportedType`) for exactly that case.
    pub fn record(self, checked: Ctx) -> void {
        for (name, kind) in checked.declared_kinds().items() {
            self.types[name] = kind
        }
    }

    #: Every recorded name paired with the type it stands for, ready to be
    #: handed to `check_program` as `outer_type_names`.
    pub fn type_names(self, u: Universe) -> list[tuple[str, Type]] {
        var out: list[tuple[str, Type]] = []
        for name in self.types.keys() {
            out.push((name, named(u, self.types[name], name)))
        }
        return out
    }
}

#: Rewrite `stmt` into `helper(expr)` so a bare expression's value reaches a
#: native that can print it. Only ever applied to a top-level expression
#: statement, which is the one place a discarded value is interesting.
fn _wrap_echo(stmt: Node, helper: str) -> Node {
    let expr = stmt.child("expr")
    if expr is null { return stmt }
    let span = expr.span
    let call = Node(
        NodeKind.Call,
        {
            "callee": Node(NodeKind.Ident, {"name": helper}, span),
            "args": [{"name": null, "value": expr, "isSpread": false, "span": span}]
        },
        span
    )
    return Node(NodeKind.ExprStmt, {"expr": call}, stmt.span)
}

#: Compile one prompt input against `session`.
#:
#: The same pipeline as `compile_source` with three differences, and each is
#: something a file never needs:
#:
#:   - the input is read by `Parser.parse_repl_line`, which decides between the
#:     two readings a prompt line has (`{"a": 1}` is a dict here and a block in
#:     a file) rather than the one a file gets;
#:   - the resolver runs in `repl_mode`, so a name an earlier line defined is a
#:     deferred global instead of E0200;
#:   - `echo`, when given, names the native a bare expression's value is handed
#:     to, so that typing `1 + 1` prints something.
#:
#: `session` is read for the type names earlier inputs declared and updated
#: with the ones this input declares -- but only when the input compiled, so a
#: rejected `class C` does not leave `C` nameable.
pub fn compile_repl(
    session: ReplSession,
    text: str,
    path: str,
    file_id: int = 0,
    opt_level: int = 2,
    echo: str = "",
    whole_file: bool = false,
    # `:disasm` compiles an input and never runs it, so what it declares is not
    # the session's.
    record: bool = true,
    # The directory imports resolve against. A prompt input's path is a label
    # like `<repl:4>`, which names no directory, so `from m import T` finds
    # nothing and `T` is not a type the next line can name. The label stays the
    # label -- it is what diagnostics and tracebacks say -- and this is only
    # where the search starts.
    source_dir: str = "",
    strict: bool = false
) -> Compiled {
    let bag = DiagnosticBag()
    let result = Compiled(path, source_hash(text), bag)

    var program: Node? = null
    if whole_file {
        program = parse(text, file_id, bag)
    } else {
        let lexer = Lexer(text, file_id, bag)
        let parser = Parser(lexer)
        let (stmt, _) = parser.parse_repl_line()
        if not lexer.had_error and stmt is not null {
            var top = stmt
            result.repl_expression = top.kind == NodeKind.ExprStmt
            if result.repl_expression and echo != "" {
                top = _wrap_echo(top, echo)
            }
            program = Node(NodeKind.Program, {"stmts": [top]}, top.span)
        }
    }
    if program == null { return result }
    result.program = program

    let resolution = resolve(program, bag, true)
    result.resolution = resolution

    let universe = Universe()
    let search_from = if source_dir == "" { path } else { f"{source_dir}/{path}" }
    let checked = check_program(
        universe, program, resolution, bag, opt_level > 0, search_from,
        session.type_names(universe), strict
    )
    result.types = checked

    let body = emit(
        program,
        resolution,
        bag,
        false,
        _REPL_MODULE_NAME,
        opt_level,
        optimize,
        text,
        checked.enum_decls()
    )
    result.body = body
    _drop_prompt_noise(bag)
    if not result.ok() { return result }

    let mod = Module(path, result.hash, body, FLAG_DEBUG)
    for name in _exported_names(body) { mod.exports.push(name) }
    result.mod = mod
    result.image = write_module(mod)
    if record { session.record(checked) }
    return result
}

# Two warnings make sense for a file and not for a prompt: at a prompt every
# binding is unused until the next line uses it, and every import is there to
# be used interactively. The C driver dropped the same two on the way out of
# `replFlushDiags`; they are dropped here instead, because the prompt now sees
# diagnostics only as the string `Compiled.report` renders.
fn _drop_prompt_noise(bag: DiagnosticBag) -> void {
    var kept: list[Diagnostic] = []
    var warnings = 0
    for diagnostic in bag.items {
        if diagnostic.code == SemaCode.W0100_UNUSED_IMPORT or
            diagnostic.code == SemaCode.W0101_UNUSED_BINDING {
            continue
        }
        if diagnostic.severity == Severity.Warning { warnings += 1 }
        kept.push(diagnostic)
    }
    bag.items = kept
    bag.warning_count = warnings
}

# Every input of one session compiles as this module, so a traceback names the
# prompt rather than the file the input happened to be labelled with.
const _REPL_MODULE_NAME = "__repl__"

#: Check the imports reachable from `path` for a cycle.
#:
#: A whole-program question, and separate from `compile_source` for that
#: reason: a module in a cycle compiles perfectly well on its own, and it is
#: only the graph the files form together that is wrong. The C splits it the
#: same way — `jaiCheckFile` runs `checkImportCycles` before it calls the front
#: end, and the compile path never runs it — so a `check` runs both and a
#: `build` runs only the second.
#:
#: The answer is the bag it reported into, so a caller with diagnostics of its
#: own can pass one in and keep everything in source order.
#:
#: A path that cannot be read is not an error here: a missing module is E0800
#: at import time and a missing entry file is the driver's to complain about.
#:
#: ```
#: let dir = __prim__.os_env("TMPDIR") ?? "/tmp"
#: for (name, body) in [("x", "from .jaithon_cycle_y import b\n"),
#:                      ("y", "from .jaithon_cycle_x import a\n")] {
#:     let handle = __prim__.io_open(f"{dir}/jaithon_cycle_{name}.jai", "wb")
#:     __prim__.io_write(handle, body)
#:     __prim__.io_close(handle)
#: }
#: let found = import_cycles(f"{dir}/jaithon_cycle_x.jai")
#: found.has_errors()                                  # true
#: found.items[0].code                                 # "E0801"
#: import_cycles("no/such/entry.jai").has_errors()     # false
#: ```
pub fn import_cycles(path: str, bag: DiagnosticBag? = null) -> DiagnosticBag {
    let into = bag ?? DiagnosticBag()
    check_import_cycles(into, path)
    return into
}

#: The name the module body carries, derived from `path`.
#:
#: `compile_source` is handed a path and not a module name, so it applies the
#: rule the C driver applies in `moduleNameFor` (src/cli/main.c): the basename
#: with `.jai` removed, or `__main__` when nothing is left. The two have to
#: agree, because the name is a constant in the record's pool and a cached
#: module whose name has lost its package cannot resolve its own exports.
#:
#: ```
#: module_name_for("lib/std/list.jai")    # "list"
#: module_name_for("<doc>")               # "<doc>"
#: module_name_for("")                    # "__main__"
#: ```
pub fn module_name_for(path: str) -> str {
    var base = path
    let cut = base.rfind("/")
    if cut >= 0 { base = base[cut + 1:] }
    if base.ends_with(".jai") { base = base[:base.len() - 4] }
    if base == "" { return "__main__" }
    return base
}

#: Compile the file at `path`.
#:
#: Raises `FileNotFoundError` when the file cannot be opened: a missing input
#: is the caller's mistake, not a diagnostic about the program.
pub fn compile_file(path: str, release: bool = false, opt_level: int = 2) -> Compiled {
    return compile_source(read_text(path), path, release, 0, opt_level)
}

#: Compile `source_path` and write its `.jaic` image to `out_path`.
#:
#: Nothing is written when compilation failed, so a stale image is never left
#: looking fresh.
pub fn build_file(
    source_path: str,
    out_path: str,
    release: bool = false,
    opt_level: int = 2
) -> Compiled {
    let built = compile_file(source_path, release, opt_level)
    if built.ok() { write_bytes(out_path, built.image) }
    return built
}

#: Read a whole file as text.
#:
#: `io_open` hands back a `file` and raises `FileNotFoundError` itself when the
#: path cannot be opened, and `io_read` always answers raw bytes whatever the
#: mode (spec §10), so the decode is here rather than in the primitive.
pub fn read_text(path: str) -> str {
    let handle = __prim__.io_open(path, "rb")
    let raw = __prim__.io_read(handle, -1)
    __prim__.io_close(handle)
    return __prim__.str_decode(raw)
}

#: Write raw bytes to a file, replacing whatever was there.
pub fn write_bytes(path: str, data: list[int]) -> void {
    let handle = __prim__.io_open(path, "wb")
    __prim__.io_write(handle, __prim__.bytes_new(data))
    __prim__.io_flush(handle)
    __prim__.io_close(handle)
}

# Every global the emitter marked with OP_EXPORT is part of the module's
# interface, and the container records the same list (§7).
fn _exported_names(body: FuncProto) -> list[str] {
    var names: list[str] = []
    var offset = 0
    while offset < body.chunk.code.len() {
        let step = disassemble_instruction(body.chunk, offset)
        if step[1] <= offset { break }
        if op_name(body.chunk.code[offset]) == "OP_EXPORT" {
            let index = _constant_operand(body, offset)
            if index < body.chunk.constants.len() {
                let value = body.chunk.constants[index]
                if value.tag == ConstTag.Str and value.value not in names {
                    names.push(value.value)
                }
            }
        }
        offset = step[1]
    }
    return names
}

fn _constant_operand(body: FuncProto, offset: int) -> int {
    let code = body.chunk.code
    return code[offset + 1] | code[offset + 2] << 8 | code[offset + 3] << 16
}

#: Disassemble a source file with this front end.
#:
#: This is what `jaithon disasm` runs when it is asked for the self-hosted
#: listing rather than the C one.
pub fn disassemble_file(path: str) -> str {
    let built = compile_file(path)
    let body = built.body
    if body == null { return built.report() }
    if not built.ok() { return built.report() + "\n" + disassemble(body, path) }
    return disassemble(body, path)
}
