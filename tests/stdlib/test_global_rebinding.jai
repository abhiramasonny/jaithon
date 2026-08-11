#: Rebinding a global must retire whatever the compiled tier cached about it.
#:
#: The compiled form of a function that calls through a module-level variable
#: bakes in what that variable held when it was compiled. Rebinding the name has
#: to invalidate that, and the module's version counter is what says so. When the
#: counter stopped moving for the case that mattered, a hot function kept calling
#: the *old* binding forever: the case below returned 4 instead of 1000003,
#: silently and with no error.
#:
#: The shape matters and is why this is a whole file rather than one assertion.
#: `caller` has to be hot enough to compile, the rebinding has to happen after
#: that, and the call after the rebinding has to be hot enough to keep running
#: compiled — a single call would fall back to the interpreter and get the right
#: answer for the wrong reason.
from std.test import assert_eq

fn _one() -> int { return 1 }

fn _two() -> int { return 1000000 }

var _bound = _one

fn _caller(n: int) -> int {
    var s = 0
    var i = 0
    while i < n {
        s = s + i
        i = i + 1
    }
    return s + _bound()
}

fn test_a_rebound_global_retires_the_compiled_form() -> void {
    # Warm `_caller` until the tier compiles it against `_one`.
    var warm = 0
    var k = 0
    while k < 400 {
        warm = warm + _caller(3)
        k = k + 1
    }
    assert_eq(_caller(3), 4, "before rebinding: 0+1+2 plus _one()")

    _bound = _two

    # A long call keeps the compiled form hot across the rebinding, so the
    # answer below comes from compiled code rather than from a fallback.
    let _hot = _caller(2_000_000)

    assert_eq(_caller(3), 1000003, "after rebinding: 0+1+2 plus _two()")
}

#: The same thing for a value, not a callee: a global an expression reads
#: directly must be re-read after it changes, never baked.
var _scale = 2

fn _scaled(n: int) -> int {
    var total = 0
    var i = 0
    while i < n {
        total = total + i * _scale
        i = i + 1
    }
    return total
}

fn test_a_rebound_value_global_is_re_read() -> void {
    var warm = 0
    var k = 0
    while k < 400 {
        warm = warm + _scaled(3)
        k = k + 1
    }
    assert_eq(_scaled(3), 6, "0*2 + 1*2 + 2*2")

    _scale = 10
    let _hot = _scaled(200_000)

    assert_eq(_scaled(3), 30, "0*10 + 1*10 + 2*10")
}
