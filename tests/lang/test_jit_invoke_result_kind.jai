#: `obj.method()` inside a compiled body when the method itself has not
#: compiled, and the kind it returns therefore has to be predicted.
#:
#: The tier used to refuse any such site outright ("if mfn->jitFunc == NULL
#: return false"), which is not a rare case at all: two methods that call each
#: other can never take turns being the first to compile, and neither can a
#: method the tier declines for some unrelated reason. The prediction now comes
#: from what the callee has been OBSERVED to return, and like every other
#: prediction in this tier it is only sound because of the guards emitted around
#: it -- the result's tag, and for an instance its object type and class shape.
#:
#: Every callee below carries a `try`, which puts OP_GET_EXC in its chunk; that
#: opcode has no JIT arm, so the method is declined as a whole and `jitFunc`
#: stays NULL however hot it gets. That is what keeps these tests aimed at the
#: predicted path rather than at the direct-call one.
#:
#: Each callee turns bad only from the FIFTH element on, which matters: a loop
#: enters its compiled form on a BACK EDGE, so a value that went wrong on
#: iteration zero would raise before the compiled body ever ran and the test
#: would pass for the wrong reason.
#:
#: The answers here are the interpreter's, so `JAITHON_NO_JIT=1` proves the
#: expectations and the plain run proves the compiled code agrees. Under
#: `JAITHON_JIT_DEOPT_STRESS=1` every one of those guards fires on every
#: iteration, which is what turns a wrong resume point into a wrong answer:
#: the call has already happened when the guard is reached, so a deopt that
#: resumed AT the call rather than after it would run it twice.
from std.test import assert_eq, assert_near, assert_throws

class Box {
    pub var n: int = 0

    fn init(self, n: int) { self.n = n }
}

class Maker {
    #: 0 -> always an instance; 1 -> a string, 2 -> an int, 3 -> a float, each
    #: only once the loop has had time to enter its compiled form.
    pub var mode: int = 0
    pub var calls: int = 0

    #: The `try` is what stops this compiling; the branch is never taken.
    pub fn make(self, i: int) -> any {
        try {
            if i < 0 { throw ValueError("unreachable") }
        } catch e: ValueError {
            return null
        }
        if i >= 4 {
            if self.mode == 1 { return "wrong" }
            if self.mode == 2 { return 5 }
            if self.mode == 3 { return 0.5 }
        }
        return Box(i)
    }

    #: `-> void`, so a caller that drops the result needs no prediction at all.
    pub fn bump(self) -> void {
        try {
            self.calls = self.calls + 1
        } catch e: ValueError {
            self.calls = 0
        }
    }

    pub fn label(self, i: int) -> any {
        try {
            if i < 0 { throw ValueError("unreachable") }
        } catch e: ValueError {
            return ""
        }
        if i >= 4 and self.mode == 1 { return 9 }
        return "abcd"
    }
}

#: Reads a field off the returned instance, so the prediction has to carry the
#: class shape and not merely "some object".
fn sum_boxes(m: Maker, n: int) -> any {
    var t = 0
    for i in 0..n {
        let b = m.make(i)
        t += b.n
    }
    return t
}

#: The result is a string, held raw -- compared, which is all SLOT_OBJ allows.
fn sum_labels(m: Maker, n: int) -> any {
    var t = 0
    for i in 0..n {
        let s = m.label(i)
        let same = s == "abcd"
        if same { t += 4 }
    }
    return t
}

#: The result is dropped by the very next instruction. Nothing is predicted and
#: nothing is guarded, so what this pins is that the call happens exactly once
#: per iteration and the operand stack is left where the loop expects it.
fn drive(m: Maker, n: int) -> void {
    for _i in 0..n { m.bump() }
}

#: Long enough for the sampling tier to compile the loop. Hotness is SIGPROF
#: ticks on back edges, not iterations, so this is a duration; nothing here
#: asserts a compile happened, and the test is correct without one.
fn warm_boxes(m: Maker) -> void {
    for _r in 0..8000 { sum_boxes(m, 8) }
}

fn warm_labels(m: Maker) -> void {
    for _r in 0..8000 { sum_labels(m, 8) }
}

fn warm_drive(m: Maker) -> void {
    for _r in 0..8000 { drive(m, 8) }
}

fn test_an_instance_result_carries_its_shape() -> void {
    let m = Maker()
    warm_boxes(m)
    assert_eq(sum_boxes(m, 8), 28)
    assert_eq(sum_boxes(m, 64), 2016)
}

fn test_a_string_where_an_instance_was_predicted_raises() -> void {
    #: The tag guard alone is not enough: VAL_OBJ is every heap object, so
    #: without the object-type check the shape guard reads `klass` one word past
    #: an ObjString's header and dereferences whatever is there.
    let m = Maker()
    warm_boxes(m)
    assert_eq(sum_boxes(m, 8), 28)
    m.mode = 1
    assert_throws(AttributeError, || sum_boxes(m, 64))
    assert_throws(AttributeError, || sum_boxes(m, 64))
}

fn test_an_int_where_an_instance_was_predicted_raises() -> void {
    let m = Maker()
    warm_boxes(m)
    assert_eq(sum_boxes(m, 8), 28)
    m.mode = 2
    assert_throws(AttributeError, || sum_boxes(m, 64))
}

fn test_a_float_where_an_instance_was_predicted_raises() -> void {
    #: A non-object where an object was predicted: the tag guard is the only
    #: thing standing between this and reading a double's bits as a pointer.
    let m = Maker()
    warm_boxes(m)
    assert_eq(sum_boxes(m, 8), 28)
    m.mode = 3
    assert_throws(AttributeError, || sum_boxes(m, 64))
    assert_near(0.5, 0.5)
}

fn test_a_string_result_is_held_raw_and_measured() -> void {
    let m = Maker()
    warm_labels(m)
    assert_eq(sum_labels(m, 8), 32)
    assert_eq(sum_labels(m, 64), 256)
}

fn test_an_int_where_a_string_was_predicted_compares_as_an_int() -> void {
    #: A non-object where SLOT_OBJ was predicted. `9 == "abcd"` is a legal
    #: comparison that is simply false, so a missing tag guard here is a wrong
    #: ANSWER and not a throw: the int would be held as a pointer and compared
    #: as a string.
    let m = Maker()
    warm_labels(m)
    assert_eq(sum_labels(m, 8), 32)
    m.mode = 1
    assert_eq(sum_labels(m, 64), 16)
    assert_eq(sum_labels(m, 64), 16)
}

fn test_a_dropped_void_result_calls_exactly_once_per_iteration() -> void {
    #: A resume point that re-ran the call, or an OP_POP skipped when it should
    #: not have been, shows up here as a count and nowhere else.
    let m = Maker()
    warm_drive(m)
    let before = m.calls
    drive(m, 64)
    assert_eq(m.calls - before, 64)
    drive(m, 64)
    assert_eq(m.calls - before, 128)
}
