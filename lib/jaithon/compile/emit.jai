#: Lower a resolved `jaithon.ast` tree to bytecode: the opcode set, the chunk, and
#: the emitter.
#:
#: Facade: imports and re-exports every module of the `emit/` directory.
#:
#: `Op` lists the opcodes of `src/vm/bytecode/chunk.h` in exactly that order: that order
#: *is* the wire format, so the enum's position is what a `.jaic` file stores
#: and a new opcode may only be appended.
from jaithon.compile.emit.opcode import FMT_MAX_PARTS, Op, STACK_EFFECT_VARIABLE
from jaithon.compile.emit.opcode import op_count, op_name, op_operand_bytes
from jaithon.compile.emit.opcode import op_stack_effect, opcode
from jaithon.compile.emit.spec import CATCH_ALL, ClassSpec, Const, ConstTag
from jaithon.compile.emit.spec import ExceptionEntry, FN_ASYNC, FN_GENERATOR
from jaithon.compile.emit.spec import FN_GETTER, FN_GPU_KERNEL, FN_KWREST
from jaithon.compile.emit.spec import FN_METHOD, FN_SETTER, FN_STATIC, FN_TRACE
from jaithon.compile.emit.spec import FN_VARIADIC, FieldSpec, LineEntry
from jaithon.compile.emit.spec import MAX_CONSTANTS, MethodSpec, SPEC_CLASS
from jaithon.compile.emit.spec import SPEC_ENUM, SPEC_TRAIT
from jaithon.compile.emit.chunk import Chunk, FuncProto
from jaithon.compile.emit.emitter import Emitter, emit

export { FMT_MAX_PARTS, Op, STACK_EFFECT_VARIABLE, op_count, op_name }
export { op_operand_bytes, op_stack_effect, opcode }
export { CATCH_ALL, ClassSpec, Const, ConstTag, ExceptionEntry, FN_ASYNC }
export { FN_GENERATOR, FN_GETTER, FN_GPU_KERNEL, FN_KWREST, FN_METHOD }
export { FN_SETTER, FN_STATIC, FN_TRACE, FN_VARIADIC, FieldSpec, LineEntry }
export { MAX_CONSTANTS, MethodSpec, SPEC_CLASS, SPEC_ENUM, SPEC_TRAIT }
export { Chunk, FuncProto }
export { Emitter, emit }
