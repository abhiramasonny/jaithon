#: `module.MEMBER` where the member is a scalar constant.
#:
#: The receiver of the field read is a MODULE, and OP_GET_FIELD wanted an
#: instance -- so reading `math.PI` once declined the entire enclosing
#: function. On the jaicv benchmark that was the largest refusal left after
#: the loop tier stopped throwing whole bodies away: 4.6% of all interpreted
#: work sat in `min_area_rect`, which reads `math.PI` exactly once.
#:
#: The value is never baked, only the entry's address, behind the same
#: keyVersion guard a plain global read stands on, plus a tag check at every
#: read -- a module global is assignable.
#:
#: Measured in one binary through JAITHON_JIT_MODULE_FIELD: four more bodies
#: reach the compiled tier on jaicv, none on the self-hosted compiler.
# jit-compiles: circle_area, two_constants, int_constant
from std.test import assert_eq, assert_true
import std.math as math
import std.io as io

#: Above JAI_JIT_THRESHOLD entries, so the bodies below reach the compiled tier.
let CALLS = 200

fn circle_area(r: float) -> float {
    return math.PI * r * r
}

fn two_constants(x: float) -> float {
    return math.PI * x + math.E
}

fn int_constant(n: int) -> int {
    return n + io.SEEK_END
}

#: Two different modules in one body. One table per body is all the guard
#: covers, so the second declines rather than pretend one keyVersion stands
#: for both -- it must still give the right answer.
fn two_modules(x: float) -> float {
    return math.PI * x + float(io.SEEK_SET)
}

pub fn test_a_module_constant_reads_right_from_compiled_code() -> void {
    var last = 0.0
    var call = 0
    while call < CALLS {
        last = circle_area(2.0)
        call += 1
    }
    assert_true(last > 12.566 and last < 12.567, "pi r squared")
}

pub fn test_two_constants_from_one_module() -> void {
    var last = 0.0
    var call = 0
    while call < CALLS {
        last = two_constants(1.0)
        call += 1
    }
    assert_true(last > 5.859 and last < 5.860, "pi + e")
}

pub fn test_an_int_constant() -> void {
    var last = 0
    var call = 0
    while call < CALLS {
        last = int_constant(5)
        call += 1
    }
    assert_eq(last, 7, "5 + SEEK_END")
}

pub fn test_two_modules_in_one_body_still_answer() -> void {
    var last = 0.0
    var call = 0
    while call < CALLS {
        last = two_modules(2.0)
        call += 1
    }
    assert_true(last > 6.283 and last < 6.284, "2pi + 0")
}
