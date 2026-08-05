#: The self-hosted compiler front end: source in, `.jaic` image out.
#:
#: `compile_source` runs the whole pipeline — lexer, parser, resolver, emitter,
#: container writer — through one `DiagnosticBag`, so a file's diagnostics come
#: out in source order whichever stage raised them, and hands back everything
#: each stage produced instead of printing anything itself.
#:
#: `compare_functions` is the other half of `make bootstrap`: it walks two
#: function records in the order `jaiCompareFunctions` walks them — arity, flags,
#: frame size, captures, defaults, code, constants, exception table — and reports
#: the first divergence. Two implementations of one specification are only worth
#: having if they are checked against each other. `--bootstrap-verify` runs the C
#: side of that comparison; `verify` is the Jaithon-side driver and **cannot run
#: today**, see its own doc comment.
#:
#: ```
#: let built = compile_source("let answer = 42\n", "<doc>")
#: built.ok()                       # true
#: built.image.len() > 32           # true
#: ```
#:
#: The file system is reached through `__prim__.io_open/read/write/flush/close`,
#: and byte strings through `__prim__.bytes_new`.
from jaithon.ast import Node
from jaithon.compile.emit import Const, ConstTag, FuncProto, emit, op_name
from jaithon.compile.jaic import FLAG_DEBUG, FLAG_RELEASE, Module, disassemble
from jaithon.compile.jaic import disassemble_instruction, float_bits, read_module
from jaithon.compile.jaic import source_hash, write_module
from jaithon.compile.lexer import DiagnosticBag
from jaithon.compile.opt import optimize
from jaithon.compile.parser import parse
from jaithon.compile.resolve import Resolution, resolve

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
    #: The container description, or null.
    pub var mod: Module?
    #: The serialised `.jaic` bytes; empty on failure.
    pub var image: list[int]

    #: Create an empty result for `path`.
    pub fn init(self, path: str, hash: int, diagnostics: DiagnosticBag) {
        self.path = path
        self.hash = hash
        self.diagnostics = diagnostics
        self.program = null
        self.resolution = null
        self.body = null
        self.mod = null
        self.image = []
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

    # The emitter is handed the optimiser rather than importing it: the passes
    # read `jaithon.compile.emit`, so the module that runs them cannot be
    # imported from there. See `_no_optimize` in that file.
    let body = emit(program, resolution, bag, release, module_name_for(path), opt_level, optimize)
    result.body = body
    if not result.ok() { return result }

    let flags = if release { FLAG_RELEASE } else { FLAG_DEBUG }
    let mod = Module(path, result.hash, body, flags)
    for name in _exported_names(body) { mod.exports.push(name) }
    result.mod = mod
    result.image = write_module(mod)
    return result
}

#: The name the module body carries, derived from `path`.
#:
#: `compile_source` is handed a path and not a module name, so it applies the
#: rule the C driver applies in `moduleNameFor` (src/cli/main.c): the basename
#: with `.jai` removed, or `__main__` when nothing is left. The two have to
#: agree, because `--bootstrap-verify` compiles the same file both ways and the
#: name is a constant in the record's pool.
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

#: What a differential check found.
pub class VerifyResult {
    #: The file that was checked.
    pub let path: str
    #: True when the two front ends agree.
    pub let agreed: bool
    #: The first divergence, or the reason the check could not run.
    pub let message: str?
    #: This front end's disassembly, when there is something to compare.
    pub let ours: str?
    #: The C front end's disassembly.
    pub let theirs: str?

    #: Record an outcome.
    pub fn init(
        self,
        path: str,
        agreed: bool,
        message: str? = null,
        ours: str? = null,
        theirs: str? = null
    ) {
        self.path = path
        self.agreed = agreed
        self.message = message
        self.ours = ours
        self.theirs = theirs
    }

    fn __str__(self) -> str {
        if self.agreed { return f"ok  {self.path}" }
        let reason = self.message ?? "?"
        var lines: list[str] = [f"DIVERGED  {self.path}", f"  {reason}"]
        if self.ours != null {
            lines.push("--- self-hosted front end ---")
            lines.push(self.ours)
        }
        if self.theirs != null {
            lines.push("--- C front end ---")
            lines.push(self.theirs)
        }
        return "\n".join(lines)
    }
}

#: Compile `path` with both front ends and compare the bytecode.
#:
#: The comparison walks the function records exactly as `jaiCompareFunctions`
#: does and ignores what the spec calls debug-only: names, line tables and
#: parameter names never fail a build.
#:
#: **This cannot run yet, and answers a `VerifyResult` saying so.** It needs the
#: C front end's bytecode as a `.jaic` image and there is no primitive that
#: hands one over: `__prim__.compile(source, path)` answers an `ObjClosure`
#: (`nReflectCompile`, `src/runtime/builtins_io.c`), not an image, so the
#: obvious call fails inside `_byte_list` with `AttributeError: 'fn' object has
#: no method 'len'`. Reporting that as an outcome rather than raising it is the
#: honest shape until the primitive exists.
#:
#: `make bootstrap` does not go through here. `--bootstrap-verify` calls
#: `compile_source` for the self-hosted side and does the comparison in C, which
#: is why this has been dead without anything noticing.
pub fn verify(path: str) -> VerifyResult {
    let text = read_text(path)
    let built = compile_source(text, path)
    if not built.ok() {
        let why = built.report()
        return VerifyResult(path, false, f"the self-hosted front end rejected the file:\n{why}")
    }
    let ours = built.body
    if ours == null {
        return VerifyResult(path, false, "the self-hosted front end emitted no body")
    }

    let reference = _reference_module(text, path, built.hash)
    if reference == null {
        return VerifyResult(
            path,
            false,
            "the C front end's bytecode cannot be reached from Jaithon: " +
                "`__prim__.compile` answers a closure, not a `.jaic` image. " +
                "Run `jaithon --bootstrap-verify` instead."
        )
    }

    let difference = compare_functions(ours, reference.body, "module body")
    if difference == null { return VerifyResult(path, true) }
    return VerifyResult(
        path,
        false,
        difference,
        disassemble(ours, "self-hosted"),
        disassemble(reference.body, "C")
    )
}

# The C front end's own image for this source, or null when there is no way to
# ask for one. Isolated so that `verify` reads as the comparison it is, and so
# that the day a primitive appears this is the only line that changes.
fn _reference_module(text: str, path: str, hash: int) -> Module? {
    let produced = __prim__.compile(text, path)
    if produced is not bytes { return null }
    return read_module(_byte_list(produced), hash)
}

#: Compare two function records, returning the first difference or null.
#:
#: `scope` names the record in the message: nested functions extend it, so a
#: divergence deep in a class reads as
#: `module body > constant 4 (Account) > constant 1 (deposit)`.
pub fn compare_functions(ours: FuncProto, theirs: FuncProto, scope: str) -> str? {
    if ours.arity != theirs.arity {
        return f"{scope}: arity differs: {ours.arity} vs {theirs.arity}"
    }
    if ours.flags != theirs.flags {
        return f"{scope}: flags differ: {ours.flags} vs {theirs.flags}"
    }
    if ours.max_slots != theirs.max_slots {
        return f"{scope}: frame size differs: {ours.max_slots} vs {theirs.max_slots}"
    }
    if ours.upvalue_count != theirs.upvalue_count {
        return f"{scope}: capture count differs: {ours.upvalue_count} vs " +
            f"{theirs.upvalue_count}"
    }
    if ours.default_offsets.len() != theirs.default_offsets.len() {
        return f"{scope}: default count differs: {ours.default_offsets.len()} vs " +
            f"{theirs.default_offsets.len()}"
    }
    for (index, offset) in ours.default_offsets.enumerate() {
        if offset != theirs.default_offsets[index] {
            return f"{scope}: default thunk {index} starts at {offset} vs " +
                f"{theirs.default_offsets[index]}"
        }
    }

    let code = _compare_code(ours, theirs, scope)
    if code != null { return code }

    if ours.chunk.constants.len() != theirs.chunk.constants.len() {
        return f"{scope}: constant count differs: {ours.chunk.constants.len()} vs " +
            f"{theirs.chunk.constants.len()}"
    }
    for (index, value) in ours.chunk.constants.enumerate() {
        let difference = _compare_constant(value, theirs.chunk.constants[index], scope, index)
        if difference != null { return difference }
    }

    if ours.chunk.exceptions.len() != theirs.chunk.exceptions.len() {
        return f"{scope}: exception table differs: {ours.chunk.exceptions.len()} vs " +
            f"{theirs.chunk.exceptions.len()} entries"
    }
    for (index, entry) in ours.chunk.exceptions.enumerate() {
        let other = theirs.chunk.exceptions[index]
        if entry.start != other.start or
            entry.end != other.end or
            entry.handler != other.handler or
            entry.type_const != other.type_const {
            return f"{scope}: exception entry {index} differs: " +
                f"[{entry.start},{entry.end})->{entry.handler} " +
                f"type {entry.type_const} vs [{other.start},{other.end})->" +
                f"{other.handler} type {other.type_const}"
        }
    }
    return null
}

fn _compare_code(ours: FuncProto, theirs: FuncProto, scope: str) -> str? {
    let mine = ours.chunk.code
    let other = theirs.chunk.code
    let common = min(mine.len(), other.len())
    var index = 0
    while index < common and mine[index] == other[index] { index += 1 }

    if index == common {
        if mine.len() == other.len() { return null }
        return f"{scope}: code length differs: {mine.len()} vs {other.len()} bytes, " +
            f"identical up to offset {index}"
    }

    let at = _instruction_start(ours, index)
    let name_ours = _op_at(ours, at)
    let name_theirs = _op_at(theirs, at)
    if name_ours != name_theirs {
        return f"{scope}: code differs at offset {at}: {name_ours} vs {name_theirs}"
    }
    return f"{scope}: code differs at offset {at}: {name_ours} operand byte " +
        f"{index - at - 1} is {mine[index]} vs {other[index]}"
}

# Start of the instruction containing byte `target`, by decoding from zero.
fn _instruction_start(proto: FuncProto, target: int) -> int {
    var offset = 0
    while offset < proto.chunk.code.len() {
        let step = disassemble_instruction(proto.chunk, offset)
        if step[1] <= offset { return target }
        if target < step[1] { return offset }
        offset = step[1]
    }
    return target
}

fn _op_at(proto: FuncProto, offset: int) -> str {
    if offset < 0 or offset >= proto.chunk.code.len() { return "<end of code>" }
    return op_name(proto.chunk.code[offset])
}

fn _compare_constant(ours: Const, theirs: Const, scope: str, index: int) -> str? {
    if ours.tag != theirs.tag {
        return f"{scope}: constant {index} is {ours.tag} here and {theirs.tag} there"
    }
    match ours.tag {
        ConstTag.Func => {
            let name = ours.value.name ?? "anonymous"
            return compare_functions(ours.value, theirs.value, f"{scope} > constant {index} ({name})")
        }
        ConstTag.ClassSpec => { return _compare_class(ours.value, theirs.value, scope, index) }
        ConstTag.Tuple => {
            if ours.value.len() != theirs.value.len() {
                return f"{scope}: constant {index} differs: tuple of " +
                    f"{ours.value.len()} vs {theirs.value.len()}"
            }
            for (position, item) in ours.value.enumerate() {
                let difference = _compare_constant(
                    item,
                    theirs.value[position],
                    f"{scope} > constant {index}",
                    position
                )
                if difference != null { return difference }
            }
            return null
        }
        ConstTag.Float => {
            # Bit equality, so that -0.0 and 0.0 count as a divergence and two
            # NaNs with the same payload do not.
            if float_equal(ours.value, theirs.value) { return null }
            return f"{scope}: constant {index} differs: {ours.value} vs {theirs.value}"
        }
        _ => {
            if ours.value == theirs.value { return null }
            return f"{scope}: constant {index} differs: {repr(ours.value)} vs " + f"{repr(theirs.value)}"
        }
    }
}

fn _compare_class(ours: any, theirs: any, scope: str, index: int) -> str? {
    if ours.name != theirs.name {
        return f"{scope}: constant {index} names class {ours.name} vs {theirs.name}"
    }
    if ours.super_name != theirs.super_name {
        return f"{scope}: class {ours.name} extends {ours.super_name} vs " + f"{theirs.super_name}"
    }
    if ours.fields.len() != theirs.fields.len() {
        return f"{scope}: class {ours.name} has {ours.fields.len()} vs " +
            f"{theirs.fields.len()} fields"
    }
    for (position, field) in ours.fields.enumerate() {
        let other = theirs.fields[position]
        if field.name != other.name or
            field.visibility != other.visibility or
            field.is_static != other.is_static {
            return f"{scope}: class {ours.name} field {position} differs: " +
                f"{field.name} vs {other.name}"
        }
    }
    if ours.methods.len() != theirs.methods.len() {
        return f"{scope}: class {ours.name} has {ours.methods.len()} vs " +
            f"{theirs.methods.len()} methods"
    }
    for (position, method) in ours.methods.enumerate() {
        let other = theirs.methods[position]
        if method.name != other.name or
            method.flags != other.flags or
            method.func_index != other.func_index {
            return f"{scope}: class {ours.name} method {position} differs: " +
                f"{method.name} vs {other.name}"
        }
    }
    return null
}

#: True when two floats have the same IEEE-754 bit pattern.
#:
#: Not `==`: that says `-0.0 == 0.0` and that no NaN equals itself, and a
#: constant pool needs neither.
#:
#: ```
#: float_equal(0.0, -0.0)      # false
#: float_equal(1.5, 1.5)       # true
#: ```
pub fn float_equal(left: float, right: float) -> bool {
    return float_bits(left) == float_bits(right)
}

fn _byte_list(data: any) -> list[int] {
    var out: list[int] = []
    for index in range(data.len()) { out.push(data[index]) }
    return out
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
