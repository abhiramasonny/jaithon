#: OP_GET_FIELD when the receiver is SLOT_INST with a pinned class but no
#: live sample to classify the field from -- what a helper call chained
#: straight into a scalar field read looks like once compiled
#: (`self._peek().kind`, `self._chunk().depth`, and the like throughout the
#: self-hosted front end). The receiver's class comes from the callee's
#: predicted return (observedReturnKind's shape, or a compiled callee's own
#: jitReturnKind/jitReturnShape contract), never from a sampled instance, so
#: pushValue leaves `seen` at NULL_VAL and the tier used to decline the whole
#: enclosing function outright with a bare "OP_GET_FIELD" -- no chain, no
#: named reason.
#:
#: A scalar field's DECLARED kind (FieldInfo::typeId, OP_FIELD_DEF's bits
#: 4-7) substitutes for the missing sample. What makes trusting it sound
#: rather than a leap is the same as every other prediction in this tier: the
#: tag is checked again at the load, and a wrong record deopts to this
#: instruction with the receiver untouched rather than answering -- exactly
#: what JAITHON_JIT_DEOPT_STRESS=1 below forces on every iteration.
#:
#: Plain top-level functions rather than methods, so the marked names below
#: are the only route to this opcode shape and scripts/jit_compile_check.py
#: (which only recognises a column-0 `fn NAME(...)`) can see them; the
#: pattern itself does not care whether the predicted call was OP_INVOKE or
#: OP_CALL, both leave the same NULL_VAL `seen`. Names are deliberately
#: distinctive (`jfk_...`) rather than generic: `jaithon test` loads the
#: self-hosted front end too, and a name like `_kind` or `_depth` collides
#: with a real function in it -- confirmed the hard way while investigating
#: this fix, where `_push` turned out to be two unrelated functions of that
#: name in the self-hosted compiler (see docs/agents/cleared-compiler.md).
# jit-compiles: jfk_read_line, jfk_read_scale, jfk_read_ready, jfk_read_marks, jfk_read_label
from std.test import assert_eq, assert_near, assert_true

class JfkInner {
    pub var line: int
    pub var scale: float
    pub var ready: bool
    #: A declared `list[T]`. Unlike the three scalars it needs a SECOND guard:
    #: its tag is VAL_OBJ, which is every heap object, so proving the tag proves
    #: nothing about the header the next arm will read. Reading ObjList's count
    #: off a string is the shape that segfaulted the VM through the dict-index
    #: arm, so the arm here proves OBJ_LIST before the entry claims to be one.
    pub var marks: list[int]
    #: A declared `str`. Predicted like the list -- VAL_OBJ plus a proof of
    #: OBJ_STRING -- and additionally carries an interned empty string as a
    #: SAMPLE, because the arms downstream (`.len()`, `==` on interned pointers,
    #: the ordering leaf call) ask `stringOperand`, which reads the sample and
    #: not the kind. A string entry with no sample is a string nothing can do
    #: anything with.
    #:
    #: The probe is honest because the guard above it is real: it chooses an arm
    #: and never removes a check. The one-byte arm, for instance, still guards
    #: the length at run time, so a probe of "" cannot make it read a longer
    #: string as one byte.
    pub var label: str

    fn init(self, line: int, scale: float, ready: bool) {
        self.line = line
        self.scale = scale
        self.ready = ready
        self.marks = [7, 8, 9]
        self.label = "abc"
    }
}

class JfkOuter {
    pub var inner: JfkInner

    fn init(self, inner: JfkInner) { self.inner = inner }
}

#: Compiles well before any of the three readers below reach
#: JAI_JIT_THRESHOLD: a bare field read off a parameter is always sampled
#: (the argument IS a live instance), so by the time a reader attempts to
#: compile, this function's `jitReturnShape` is on record and
#: `jaiClassForShape` resolves it -- the direct-call route through
#: emitDirectCall, the common case this test targets.
fn jfk_inner_of(o: JfkOuter) -> JfkInner { return o.inner }

fn jfk_read_line(o: JfkOuter) -> int { return jfk_inner_of(o).line }

fn jfk_read_scale(o: JfkOuter) -> float { return jfk_inner_of(o).scale }

fn jfk_read_ready(o: JfkOuter) -> bool { return jfk_inner_of(o).ready }

#: `len`, not `marks[1]`. The arm predicts that the field IS a list; it cannot
#: predict what is IN it, and the index arm needs an element exemplar it has no
#: way to supply. Asserting on the index would be asserting on a different arm.
fn jfk_read_marks(o: JfkOuter) -> int { return len(jfk_inner_of(o).marks) }

fn jfk_read_label(o: JfkOuter) -> int { return jfk_inner_of(o).label.len() }

#: Settles jfk_inner_of into its own standalone compiled form, called from
#: OUTSIDE any of the three readers, before any of them run: a reader's own
#: warm-up loop calls jfk_inner_of once per iteration too, so without this
#: the two cross JAI_JIT_THRESHOLD in the same call and which one compiles
#: first becomes a race -- the loser sees jfk_inner_of with no jitFunc yet
#: and inlines it instead of reaching this fix's arm at all (harmless, but
#: it stops the test proving anything about THIS fix). Idempotent and cheap
#: to call from each test below, so test order does not matter either.
fn jfk_settle_inner_of(o: JfkOuter) -> void {
    for _i in 0..300 { let _ = jfk_inner_of(o) }
}

fn jfk_warm_line(o: JfkOuter) -> int {
    var t = 0
    for _i in 0..200 { t += jfk_read_line(o) }
    return t
}

fn jfk_warm_scale(o: JfkOuter) -> float {
    var t = 0.0
    for _i in 0..200 { t += jfk_read_scale(o) }
    return t
}

fn jfk_warm_ready(o: JfkOuter) -> int {
    var t = 0
    for _i in 0..200 {
        if jfk_read_ready(o) { t += 1 }
    }
    return t
}

fn jfk_warm_marks(o: JfkOuter) -> int {
    var t = 0
    for _i in 0..200 {
        t += jfk_read_marks(o)
    }
    return t
}

fn jfk_warm_label(o: JfkOuter) -> int {
    var t = 0
    for _i in 0..200 {
        t += jfk_read_label(o)
    }
    return t
}

pub fn test_str_field_off_a_predicted_receiver() -> void {
    let outer = JfkOuter(JfkInner(11, 0.5, true))
    assert_eq(jfk_warm_label(outer), 3 * 200, "warm")
    assert_eq(jfk_read_label(outer), 3, "and after")
    #: The sample is a PROBE, an empty interned string, not the real value --
    #: so anything reading it as a value rather than as a type would answer 0
    #: here instead of 3.
    assert_true(outer.inner.label == "abc", "the real value is intact")
}

pub fn test_list_field_off_a_predicted_receiver() -> void {
    let outer = JfkOuter(JfkInner(11, 0.5, true))
    assert_eq(jfk_warm_marks(outer), 3 * 200, "warm")
    assert_eq(jfk_read_marks(outer), 3, "and after")
    #: The second guard is what this asks for: swap the list for something that
    #: is also VAL_OBJ and the read must deopt rather than read a list header
    #: off it.
    assert_eq(len(outer.inner.marks), 3, "still a list of three")
}

fn test_int_field_off_a_predicted_receiver() -> void {
    let o = JfkOuter(JfkInner(7, 1.5, true))
    jfk_settle_inner_of(o)
    assert_eq(jfk_warm_line(o), 1400)
    assert_eq(jfk_read_line(o), 7)
}

fn test_float_field_off_a_predicted_receiver() -> void {
    let o = JfkOuter(JfkInner(7, 1.5, true))
    jfk_settle_inner_of(o)
    assert_near(jfk_warm_scale(o), 300.0)
    assert_near(jfk_read_scale(o), 1.5)
}

fn test_bool_field_off_a_predicted_receiver() -> void {
    let on = JfkOuter(JfkInner(7, 1.5, true))
    jfk_settle_inner_of(on)
    assert_eq(jfk_warm_ready(on), 200)
    let off = JfkOuter(JfkInner(7, 1.5, false))
    assert_eq(jfk_warm_ready(off), 0)
}
