#: The record shapes a `.jaic` carries: the pool entry, the two tables beside
#: the code, and the class spec.

#: `ObjFunction` flag bits, as `spec/BYTECODE.md` §5 stores them.
pub const FN_VARIADIC: int = 1
pub const FN_KWREST: int = 2
pub const FN_METHOD: int = 4
pub const FN_STATIC: int = 8
pub const FN_GENERATOR: int = 16
pub const FN_ASYNC: int = 32
pub const FN_GETTER: int = 64
pub const FN_SETTER: int = 128
pub const FN_TRACE: int = 512
pub const FN_GPU_KERNEL: int = 1024

#: Largest constant index a `u24` operand can hold.
pub const MAX_CONSTANTS: int = 16777216

#: `kind` of the class-spec constant `OP_CLASS` reads (`classSpecInstantiate`).
pub const SPEC_CLASS: int = 0
pub const SPEC_TRAIT: int = 1
pub const SPEC_ENUM: int = 2

#: `typeConst` of an exception-table entry that catches everything.
#: `JAI_HANDLER_CATCH_ALL` in `src/vm/vm.c`; no pool index can reach it.
pub const CATCH_ALL: int = 4294967295

#: The constant-pool tags of `spec/BYTECODE.md` §4, in tag order.
pub enum ConstTag { Null, Bool, Int, Float, Str, Bytes, Func, ClassSpec, Tuple, Type }

#: One constant-pool entry: a tag and the payload that tag implies.
pub class Const {
    #: Which of the ten payload shapes this is.
    pub let tag: ConstTag
    #: `bool`, `int`, `float`, `str`, `list[int]` of bytes, `FuncProto`,
    #: `ClassSpec`, `list[Const]` for a tuple, or a type name. Mutable because a
    #: class spec read back from a file names its own pool by index, and those
    #: indices become strings only once the pool is complete.
    pub var value: any

    #: Wrap `value` with its tag.
    pub fn init(self, tag: ConstTag, value: any) {
        self.tag = tag
        self.value = value
    }
}

#: One line-table row: where an instruction starts and the source it came from.
pub class LineEntry {
    #: Code offset of the first instruction covered.
    pub let offset: int
    #: Byte offset of the start of the source span.
    pub let span_start: int
    #: Byte offset one past the end of the source span.
    pub let span_end: int

    #: Record a row.
    pub fn init(self, offset: int, span_start: int, span_end: int) {
        self.offset = offset
        self.span_start = span_start
        self.span_end = span_end
    }
}

#: One row of a function's exception table: the guarded region and its handler.
pub class ExceptionEntry {
    #: First code offset covered.
    pub var start: int
    #: One past the last code offset covered.
    pub var end: int
    #: Where the handler begins.
    pub var handler: int
    #: Constant index of the caught class, or 0 for catch-all.
    pub var type_const: int

    #: Record a region.
    pub fn init(self, start: int, end: int, handler: int, type_const: int) {
        self.start = start
        self.end = end
        self.handler = handler
        self.type_const = type_const
    }
}

#: One field of a class spec (`spec/BYTECODE.md` §6).
pub class FieldSpec {
    #: The field name.
    pub let name: str
    #: 0 private, 1 protected, 2 public.
    pub let visibility: int
    #: True for a class-level field, which occupies no instance slot.
    pub let is_static: bool
    #: Constant index of the declared type, or 0 when untyped.
    pub let type_index: int

    #: Describe one field.
    pub fn init(self, name: str, visibility: int, is_static: bool, type_index: int) {
        self.name = name
        self.visibility = visibility
        self.is_static = is_static
        self.type_index = type_index
    }
}

#: One method of a class spec (`spec/BYTECODE.md` §6).
pub class MethodSpec {
    #: The method name.
    pub let name: str
    #: 0 private, 1 protected, 2 public.
    pub let visibility: int
    #: `FN_*` bits: method, static, getter, setter, variadic, kwrest.
    pub let flags: int
    #: Index of the method's function record in the enclosing pool.
    pub let func_index: int

    #: Describe one method.
    pub fn init(self, name: str, visibility: int, flags: int, func_index: int) {
        self.name = name
        self.visibility = visibility
        self.flags = flags
        self.func_index = func_index
    }
}

#: A class as the constant pool stores it, from which `OP_CLASS` builds one.
pub class ClassSpec {
    #: The class name.
    pub let name: str
    #: The superclass name, or null.
    pub let super_name: str?
    #: Names of the implemented traits.
    pub let traits: list[str]
    #: Fields in declaration order; slots are assigned parents first.
    pub let fields: list[FieldSpec]
    #: Methods, static methods, getters and setters.
    pub let methods: list[MethodSpec]

    #: Describe one class.
    pub fn init(
        self,
        name: str,
        super_name: str?,
        traits: list[str],
        fields: list[FieldSpec],
        methods: list[MethodSpec]
    ) {
        self.name = name
        self.super_name = super_name
        self.traits = traits
        self.fields = fields
        self.methods = methods
    }
}
