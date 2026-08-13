#: `x.len()` compiled to a load from the receiver instead of a call.
#:
#: The compiled tier replaces a whole invoke with one instruction when the
#: builtin's body is a field read (`src/vm/jit/jit_field_read.h`): `k.len()`
#: was 39 arm64 instructions and a `blr` into jitInvokeNative to fetch a 32-bit
#: field. `tests/vm/field_natives.c` checks the TABLE -- that the offset and the
#: result kind agree with what the native returns. This checks the EMISSION:
#: that the type guard in front of the load is real, that a string's lazily
#: computed scalar count is answered rather than read from its sentinel, and
#: that a receiver whose type changes under a compiled body still gets the
#: interpreter's answer.
#:
#: One counting function PER RECEIVER TYPE, deliberately: the tier specialises
#: a body to the type it observed, so a single shared function would compile
#: for whichever type warmed it first and every other case would deopt and run
#: interpreted -- a test that reads as coverage while exercising nothing. Each
#: of the six below was confirmed to reach `[jit] compiled ...` with
#: `JAI_JIT_WHY=1`.
#:
#: The expectations are the interpreter's, so `JAITHON_NO_JIT=1` proves them and
#: the plain run proves the compiled code agrees. Under
#: `JAITHON_JIT_DEOPT_STRESS=1` every guard fires on every iteration, which is
#: the pass that catches a resume point that consumed the receiver.
from std.test import assert_eq, assert_throws

fn count_str(x: any) -> int {
    var total = 0
    var i = 0
    while i < 200 {
        total += x.len()
        i += 1
    }
    return total
}

fn count_list(x: any) -> int {
    var total = 0
    var i = 0
    while i < 200 {
        total += x.len()
        i += 1
    }
    return total
}

fn count_dict(x: any) -> int {
    var total = 0
    var i = 0
    while i < 200 {
        total += x.len()
        i += 1
    }
    return total
}

fn count_set(x: any) -> int {
    var total = 0
    var i = 0
    while i < 200 {
        total += x.len()
        i += 1
    }
    return total
}

fn count_tuple(x: any) -> int {
    var total = 0
    var i = 0
    while i < 200 {
        total += x.len()
        i += 1
    }
    return total
}

fn count_bytes(x: any) -> int {
    var total = 0
    var i = 0
    while i < 200 {
        total += x.len()
        i += 1
    }
    return total
}

#: Warmed on a string and then handed everything else, which is the one case
#: that must NOT answer from the load.
fn count_any(x: any) -> int {
    var total = 0
    var i = 0
    while i < 200 {
        total += x.len()
        i += 1
    }
    return total
}

#: The result discarded rather than used, which the arm handles differently: it
#: pushes the length and leaves the following OP_POP to be walked.
fn touch(x: any) -> int {
    var i = 0
    while i < 200 {
        x.len()
        i += 1
    }
    return i
}

fn test_len_of_a_string_reads_the_scalar_count() -> void {
    let s = "abcde"
    var i = 0
    while i < 200 {
        count_str(s)
        i += 1
    }
    assert_eq(count_str(s), 1000)
    #: Five scalars in ten bytes: a load from `length` instead of `scalars`
    #: doubles the answer.
    assert_eq(count_str("ééééé"), 1000)
}

fn test_len_of_a_list_reads_the_count_not_the_capacity() -> void {
    let xs = [1, 2, 3, 4, 5, 6, 7]
    var i = 0
    while i < 200 {
        count_list(xs)
        i += 1
    }
    assert_eq(count_list(xs), 1400)
    #: A list built by pushing has spare capacity, so a load from `capacity`
    #: reads high here and the two answers separate.
    var grown = []
    var j = 0
    while j < 7 {
        grown.push(j)
        j += 1
    }
    assert_eq(count_list(grown), 1400)
}

fn test_len_of_a_dict_reads_into_its_table() -> void {
    let d = {"a": 1, "b": 2, "c": 3}
    var i = 0
    while i < 200 {
        count_dict(d)
        i += 1
    }
    assert_eq(count_dict(d), 600)
}

fn test_len_of_a_set_reads_into_its_table() -> void {
    var st = set()
    st.add(1)
    st.add(2)
    var i = 0
    while i < 200 {
        count_set(st)
        i += 1
    }
    assert_eq(count_set(st), 400)
}

fn test_len_of_a_tuple_and_of_bytes() -> void {
    let t = (1, 2, 3, 4)
    var i = 0
    while i < 200 {
        count_tuple(t)
        i += 1
    }
    assert_eq(count_tuple(t), 800)

    #: `str.bytes()` returns a LIST of byte values, not a `bytes` object -- the
    #: first version of this test used it and exercised the list arm twice.
    let b = bytes("hello")
    var j = 0
    while j < 200 {
        count_bytes(b)
        j += 1
    }
    assert_eq(count_bytes(b), 1000)
}

fn test_a_lazily_counted_string_is_still_counted() -> void {
    #: ObjString::scalars holds UINT32_MAX until someone asks, and the emitted
    #: fast path must NOT answer from the sentinel -- 4294967295 is what a
    #: missing sentinel test would add. Each f-string below is freshly built,
    #: so its memo is empty and the inline slow path is what answers.
    var i = 0
    var seen = 0
    while i < 300 {
        let fresh = f"éé{i}"
        seen += fresh.len()
        i += 1
    }
    #: Two scalars of prefix each, plus the digits of 0..299: ten one-digit,
    #: ninety two-digit, two hundred three-digit.
    assert_eq(seen, 300 * 2 + 10 * 1 + 90 * 2 + 200 * 3)
}

fn test_the_receiver_type_is_guarded_not_assumed() -> void {
    #: The load is only sound behind the type guard: a list's count and a
    #: string's scalar count live at different offsets, and a dict's count is
    #: inside its table. A body compiled for one and handed another must get
    #: the interpreter's answer, not a word read off the wrong header.
    let s = "abcde"
    var i = 0
    while i < 200 {
        count_any(s)
        i += 1
    }
    assert_eq(count_any(s), 1000)
    assert_eq(count_any([1, 2, 3]), 600)
    assert_eq(count_any({"x": 1}), 200)
    assert_eq(count_any((1, 2)), 400)
    assert_eq(count_any(bytes("hello")), 1000)
    assert_eq(count_any(s), 1000)
}

fn test_a_receiver_with_no_len_still_raises() -> void {
    let s = "abcde"
    var i = 0
    while i < 200 {
        count_any(s)
        i += 1
    }
    #: An int has no `len`, and the compiled body must reach the interpreter to
    #: say so rather than reading a field off an integer.
    assert_throws(AttributeError, || count_any(7))
    assert_throws(AttributeError, || count_any(7))
    assert_eq(count_any(s), 1000)
}

fn test_a_discarded_length_consumes_exactly_one_stack_entry() -> void {
    #: The arm pushes the result and lets the following OP_POP take it. If it
    #: consumed the OP_POP as well the walk would land mid-instruction, which
    #: is what `declined at OP_NOP` looks like -- or worse, compile with the
    #: operand stack one entry off.
    let xs = [1, 2, 3]
    var i = 0
    while i < 200 {
        touch(xs)
        i += 1
    }
    assert_eq(touch(xs), 200)
}

fn test_a_list_that_grows_under_a_compiled_length() -> void {
    #: The load reads the live field, so a push between two calls must show up.
    #: A hoisted or cached count would answer 3 forever.
    let xs = [1, 2, 3]
    var i = 0
    while i < 200 {
        count_list(xs)
        i += 1
    }
    assert_eq(count_list(xs), 600)
    xs.push(4)
    assert_eq(count_list(xs), 800)
    xs.pop()
    assert_eq(count_list(xs), 600)
}
