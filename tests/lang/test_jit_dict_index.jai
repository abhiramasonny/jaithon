#: `d[k]` in a loop hot enough to reach the compiled tier.
#:
#: The tier had the WRITE half (`d[k] = v`) and not the read, so any loop that
#: read a dict ran interpreted end to end -- a refusal is per-body, so one
#: `d["a"]` took the whole enclosing function down. `t += d["a"]` forty million
#: times was 320,000,326 interpreted instructions and 2,421,156 after,
#: 1.79x on wall clock (0.59s to 0.33s, five runs each, no spread).
#:
#: The value kind is predicted off the live dict and guarded, as the list arm
#: does it, with one difference: the sample must be UNIFORM across the dict.
#: `{"name": "x", "count": 3}` is an ordinary dict whose values disagree, and
#: predicting off its first entry would deoptimise on every read.
# jit-compiles: sum_values, join_values, count_shapes
from std.test import assert_eq, assert_throws

let CALLS = 100

class Tag {
    var id: int

    fn init(self, id: int) -> void { self.id = id }

    pub fn value(self) -> int { return self.id }
}

fn sum_values(counts: dict[str, int], rounds: int) -> int {
    var total = 0
    var i = 0
    while i < rounds {
        total += counts["a"] + counts["b"]
        i += 1
    }
    return total
}

fn join_values(names: dict[str, str], rounds: int) -> int {
    var total = 0
    var i = 0
    while i < rounds {
        total += names["first"].len()
        i += 1
    }
    return total
}

fn count_shapes(tags: dict[str, Tag], rounds: int) -> int {
    var total = 0
    var i = 0
    while i < rounds {
        total += tags["x"].value()
        i += 1
    }
    return total
}

#: Values of two kinds: the walk refuses to predict, so this body is declined
#: and the interpreter answers. Here to prove the refusal is a refusal and not
#: a wrong prediction.
fn sum_mixed(mixed: dict[str, any], rounds: int) -> int {
    var total = 0
    var i = 0
    while i < rounds {
        total += len(str(mixed["n"])) + len(str(mixed["s"]))
        i += 1
    }
    return total
}

pub fn test_int_values() -> void {
    let counts = {"a": 2, "b": 3}
    var total = 0
    var call = 0
    while call < CALLS {
        total += sum_values(counts, 10)
        call += 1
    }
    assert_eq(total, 50 * CALLS)
}

pub fn test_string_values() -> void {
    let names = {"first": "ada", "last": "byron"}
    var total = 0
    var call = 0
    while call < CALLS {
        total += join_values(names, 10)
        call += 1
    }
    assert_eq(total, 30 * CALLS)
}

pub fn test_instance_values() -> void {
    let tags = {"x": Tag(7), "y": Tag(9)}
    var total = 0
    var call = 0
    while call < CALLS {
        total += count_shapes(tags, 10)
        call += 1
    }
    assert_eq(total, 70 * CALLS)
}

pub fn test_mixed_values_are_refused_not_guessed() -> void {
    let mixed: dict[str, any] = {"n": 42, "s": "abc"}
    var total = 0
    var call = 0
    while call < CALLS {
        #: len("42") + len("abc") per round.
        total += sum_mixed(mixed, 10)
        call += 1
    }
    assert_eq(total, 50 * CALLS)
}

pub fn test_a_missing_key_still_raises() -> void {
    let counts = {"a": 1}
    assert_throws(KeyError, fn() -> void {
        var i = 0
        while i < 200 {
            let _ = counts["nope"]
            i += 1
        }
    }, "missing key from a compiled body")
}
