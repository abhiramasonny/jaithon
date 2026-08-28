#: A dict holding more than one kind of object, read from a compiled body.
#:
#: `jitGetIndexDict` predicts the value's kind by sampling a prefix of the
#: dict, and when that prefix is all instances it guards the result with a
#: shape check. The check read `instance->klass` after proving only that the
#: value was VAL_OBJ -- which every heap object is. A str under the next key
#: put its length and hash where the klass pointer belongs, and dereferencing
#: that SEGFAULTED the VM.
#:
#: Nothing exotic reaches it: `d[k]` on a `dict[str, any]` in any body hot
#: enough to compile. It surfaced only because a new `in` arm let a body in the
#: self-hosted compiler walk far enough to reach the dict read after it.
#:
#: The shared descriptor-return path had already grown this guard, with a
#: comment giving this exact reason; the dict arm was written separately and
#: did not. The tag test cannot stand in for it -- it is the same test the
#: sampled prefix passed.
from std.test import assert_eq, assert_true

class Box {
    pub var v: int

    pub fn init(self, v: int) -> void {
        self.v = v
    }
}

#: Above JAI_JIT_THRESHOLD entries, so this reaches the compiled tier and the
#: shape guard is the one that runs.
let CALLS = 400

fn probe(d: dict[str, any], k: str) -> any {
    return d[k]
}

fn uniform_dict() -> dict[str, any] {
    var d: dict[str, any] = {}
    var i = 0
    while i < 40 {
        d[str(i)] = Box(i)
        i += 1
    }
    return d
}

fn warm(d: dict[str, any]) -> int {
    var seen = 0
    var n = 0
    while n < CALLS {
        let _ = probe(d, str(n % 40))
        seen += 1
        n += 1
    }
    return seen
}

pub fn test_a_dict_of_one_shape_still_reads() -> void {
    let d = uniform_dict()
    assert_eq(warm(d), CALLS, "warm-up")
    let b = probe(d, "7")
    assert_true(b != null, "the instance came back")
}

pub fn test_a_later_key_of_another_type_does_not_crash() -> void {
    let d = uniform_dict()
    let _ = warm(d)
    #: The sampled prefix was all Box. This one is not, and the guard has to
    #: notice before it reads a klass off a string header.
    d["7"] = "a string here"
    let s = probe(d, "7")
    assert_eq(s, "a string here", "the str came back")
}

pub fn test_the_other_kinds_a_dict_can_hold() -> void {
    let d = uniform_dict()
    let _ = warm(d)
    d["1"] = [1, 2, 3]
    d["2"] = {"nested": 1}
    d["3"] = 17
    d["4"] = null
    assert_eq(len(probe(d, "1")), 3, "list")
    assert_eq(probe(d, "3"), 17, "int")
    assert_true(probe(d, "4") == null, "null")
}
