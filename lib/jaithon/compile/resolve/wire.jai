#: The one module that knows every other module of `resolve/` at once: it
#: fills in the three seams `resolver.jai` declares, and holds `resolve`, the
#: front door of the whole directory.
#:
#: It has to be a module that imports everything and is imported by nothing but
#: the facade, because a seam exists exactly where an import may not go.
#:
#: The wiring runs as a statement at module scope rather than as a call each
#: caller makes. `resolve.jai` imports this file, so anything that reaches the
#: package by its facade -- which is everything -- has already run it. The
#: sibling of `check/checker.jai`'s `install`, except that nobody has to
#: remember it.
from jaithon.ast import Node
from jaithon.compile.diag import DiagnosticBag
from jaithon.compile.symbol import Resolution
from jaithon.compile.resolve.resolver import Resolver, install_hooks
from jaithon.compile.resolve.decl import _assign_field_slots
from jaithon.compile.resolve.walk import _block_contents, _node

install_hooks(_node, _block_contents, _assign_field_slots)

#: Resolve `program`, returning the table `emit.jai` reads.
#:
#: Diagnostics go into `bag`; ask `bag.has_errors()` whether the result is
#: usable.
pub fn resolve(program: Node, bag: DiagnosticBag, repl_mode: bool = false) -> Resolution {
    return Resolver(bag, repl_mode).resolve_program(program)
}
