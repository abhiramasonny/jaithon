#: The emitter itself: the walk that lowers a resolved tree into a chunk.
#:
#: Facade: imports and re-exports the modules of the `emit/` directory. The
#: surface is two names, and it was two names before the split.
from jaithon.compile.emit.state import Emitter
from jaithon.compile.emit.wire import emit

export { Emitter, emit }
