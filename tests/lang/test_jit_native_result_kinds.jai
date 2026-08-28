#: Builtins whose result kind is a property of the function, not of its
#: arguments, called from a loop hot enough to reach the compiled tier.
#:
#: The tier used to emit a builtin call only when it could produce the answer
#: itself -- `float(i)` is one instruction, `ord(c)` is a byte load -- and
#: declined the WHOLE enclosing body for every other one. A single `str()` in a
#: loop was enough: a probe doing `str(i % 10_000)` per iteration ran 2,800,321
#: interpreted instructions, and 20,382 once the call was emitted (1.41x on
#: wall clock, the rest being the string allocation itself).
#:
#: `str` is a string whatever it is handed, `len` an int, `bool` a bool. That
#: is the whole licence: the call goes out as an ordinary call and the returned
#: tag is guarded, so a row that is ever wrong deoptimises rather than lying.
#: These check each result kind travels back correctly -- a bool especially,
#: which is ONE byte of the Value union and was read eight bytes wide once.
# jit-compiles: total_digits, name_lengths, count_truthy, count_instances, parse_all
from std.test import assert_eq, assert_throws

#: Called often enough to cross the function tier's 64-entry threshold, which
#: is a count rather than the profiling timer the loop tier waits on -- so what
#: these compile does not depend on how fast the machine running them is.
let CALLS = 100

class Point {
    var x: int

    fn init(self, x: int) -> void { self.x = x }
}

fn total_digits(rounds: int) -> int {
    var total = 0
    var round = 0
    while round < rounds {
        var index = 0
        #: 0..9 are one digit each, 10..99 are two.
        while index < 100 {
            total += len(str(index))
            index += 1
        }
        round += 1
    }
    return total
}

fn name_lengths(rounds: int) -> int {
    var total = 0
    var round = 0
    while round < rounds {
        var index = 0
        while index < 10 {
            total += len(type_of(index)) + len(repr(index)) + ord(chr(65))
            index += 1
        }
        round += 1
    }
    return total
}

fn count_truthy(values: list[int], rounds: int) -> int {
    var seen = 0
    var round = 0
    while round < rounds {
        var index = 0
        while index < values.len() {
            if bool(values[index]) { seen += 1 }
            index += 1
        }
        round += 1
    }
    return seen
}

fn count_instances(items: list[Point], rounds: int) -> int {
    var seen = 0
    var round = 0
    while round < rounds {
        var index = 0
        while index < items.len() {
            if isinstance(items[index], Point) { seen += 1 }
            index += 1
        }
        round += 1
    }
    return seen
}

fn parse_all(text: str, rounds: int) -> int {
    var total = 0
    var round = 0
    while round < rounds {
        total += int(text) + int("ff", 16)
        round += 1
    }
    return total
}

pub fn test_string_results_travel_back() -> void {
    var digits = 0
    var names = 0
    var call = 0
    while call < CALLS {
        digits += total_digits(1)
        #: Per inner step: len("int") + len(repr(i)) + ord("A"), and repr of a
        #: single-digit int is one character.
        names += name_lengths(1)
        call += 1
    }
    assert_eq(digits, 190 * CALLS)
    assert_eq(names, (3 + 1 + 65) * 10 * CALLS)
}

pub fn test_bool_results_are_one_byte() -> void {
    #: A bool read eight bytes wide picks up whatever sat beside it in the
    #: result slot, which is nonzero often enough to look like `true`.
    let values = [0, 1, 0, 2, 0]
    let points = [Point(1), Point(2)]
    var truthy = 0
    var instances = 0
    var call = 0
    while call < CALLS {
        truthy += count_truthy(values, 1)
        instances += count_instances(points, 1)
        call += 1
    }
    assert_eq(truthy, 2 * CALLS)
    assert_eq(instances, 2 * CALLS)
}

pub fn test_int_conversion_from_strings() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        total += parse_all("21", 1)
        call += 1
    }
    assert_eq(total, (21 + 255) * CALLS)
}

pub fn test_the_error_cases_still_raise() -> void {
    #: The natives raise from inside compiled code, and the interpreter has to
    #: own the exception from there.
    assert_throws(ValueError, fn() -> void { let _ = chr(-1) }, "chr range")
    assert_throws(TypeError, fn() -> void { let _ = len(5) }, "len of an int")
    assert_throws(ValueError, fn() -> void { let _ = int("nope") }, "int of a word")
}
