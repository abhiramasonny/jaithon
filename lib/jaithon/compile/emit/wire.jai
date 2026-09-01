#: The one module that knows every other module of `emit/` at once: it fills in
#: the three seams `state.jai` declares, and holds `emit`, the front door of
#: the whole directory.
#:
#: It has to be a module that imports everything and is imported by nothing but
#: the facade, because a seam exists exactly where an import may not go.
#:
#: The wiring runs as a statement at module scope rather than as a call each
#: caller makes. `emitter.jai` imports this file, so anything that can name
#: `Emitter` has already run it, and an `Emitter` built by hand keeps working
#: untouched. The sibling of `parse/wire.jai`.
from jaithon.ast import Node
from jaithon.compile.diag import DiagnosticBag
from jaithon.compile.symbol import Resolution
from jaithon.compile.emit.chunk import FuncProto
from jaithon.compile.emit.state import Emitter, _no_optimize, install_hooks
from jaithon.compile.emit.function import _closure
from jaithon.compile.emit.pattern import _match
from jaithon.compile.emit.stmt import _stmt

install_hooks(_stmt, _match, _closure)

#: Lower a resolved program and return the module body's function record.
#:
#: The record comes back even when a diagnostic was reported, so a caller can
#: disassemble what was produced while reporting what was not.
pub fn emit(
    program: Node,
    resolution: Resolution,
    bag: DiagnosticBag,
    strip_asserts: bool = false,
    module_name: str? = null,
    opt_level: int = 2,
    optimize: fn(FuncProto, int) -> void = _no_optimize,
    source: str? = null,
    imported_enums: dict[str, Node] = {}
) -> FuncProto {
    let emitter = Emitter(resolution, bag, strip_asserts, opt_level, optimize)
    emitter.source = source
    emitter.imported_enums = imported_enums
    return emitter.compile_program(program, module_name)
}
