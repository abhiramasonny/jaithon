#: A class with no explicit `init` gets one synthesised, and the bytecode it
#: emits must be the same width as the one `emitMethod` writes for a real
#: method. It was not: the synthesised path emitted the name and the flags but
#: not the visibility byte, while `chunk.c` declares OP_METHOD five operand
#: bytes wide. The VM read the following opcode as the visibility and then
#: decoded everything after it one byte out of phase, so the class never bound
#: its own name and the first use reported `NameError: undefined name`.
#:
#: Nothing in the suite declared a field-only class, which is why an entire
#: ordinary class shape could stop working without a single test noticing.
from std.test import assert_eq, assert_true

class FieldsOnly {
    pub var n: int = 1
}

class WithInit {
    pub var n: int

    pub fn init(self, v: int) {
        self.n = v
    }
}

class FieldsOnlyMultiple {
    pub var a: int = 1
    pub var b: str = "two"
    pub var c: float = 3.0
}

fn test_a_class_without_an_explicit_init_binds_its_own_name() {
    #: The original symptom. Constructing it at all is the assertion.
    let value = FieldsOnly()
    assert_eq(value.n, 1)
}

fn test_a_synthesised_init_applies_every_field_default() {
    let value = FieldsOnlyMultiple()
    assert_eq(value.a, 1)
    assert_eq(value.b, "two")
    assert_eq(value.c, 3.0)
}

fn test_an_explicit_init_still_works_beside_a_synthesised_one() {
    #: Both emit OP_METHOD. If the two paths ever disagree on width again, one
    #: of these two constructions decodes the other's bytes.
    assert_eq(FieldsOnly().n, 1)
    assert_eq(WithInit(7).n, 7)
}

fn test_declarations_after_a_field_only_class_are_reachable() {
    #: The phase error corrupted the instruction stream *after* the class, so a
    #: later declaration is what actually disappeared. Naming both proves the
    #: stream stayed aligned.
    assert_true(FieldsOnly() != null)
    assert_true(WithInit(1) != null)
}
