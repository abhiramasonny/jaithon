#: A result the tier produced itself, used as a receiver one instruction later.
#:
#: An f-string's result and a builtin method's result do not exist until the
#: compiled code runs, so there is no Value for the model to sample -- and the
#: invoke arm resolves WHICH builtin a name means from the receiver's sample.
#: So `f"{a}-{b}".len()` and `s.lower().len()` declined the whole enclosing
#: loop at the second call, one instruction after the tier had just learned to
#: emit the first.
#:
#: The type is known even when the value is not: jitFormat always builds a
#: string, and an invoke site's InlineCache::resultKind names the ObjType it
#: returned. Recording that (Emit::stackObjType) lets the arm resolve the
#: method through an empty probe string, exactly as the list arm already builds
#: an empty probe list for a list it has no sample of. It is a PREDICTION -- the
#: receiver's Obj.type is guarded at run time either way -- so it survives a
#: branch merge and can only ever cost a deopt.
#:
#: `s.lower().len()` a million times: 8,000,322 interpreted instructions before
#: and 10,152 after (1.67x). The f-string form: 10,000,320 and 32,668 (1.10x,
#: the rest being the string the f-string has to allocate).
# jit-compiles: chain_methods, format_then_measure, format_then_compare
from std.test import assert_eq

let CALLS = 100

fn chain_methods(text: str, rounds: int) -> int {
    var total = 0
    var i = 0
    while i < rounds {
        total += text.lower().len() + text.upper().len()
        i += 1
    }
    return total
}

fn format_then_measure(rounds: int) -> int {
    var total = 0
    var i = 0
    while i < rounds {
        total += f"{i}-{i}".len()
        i += 1
    }
    return total
}

fn format_then_compare(rounds: int) -> int {
    var seen = 0
    var i = 0
    while i < rounds {
        if f"n{i}".starts_with("n") { seen += 1 }
        i += 1
    }
    return seen
}

pub fn test_a_method_result_is_a_receiver() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        total += chain_methods("Hello", 10)
        call += 1
    }
    assert_eq(total, 100 * CALLS)
}

pub fn test_an_fstring_is_a_receiver() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        #: "0-0" through "9-9": three characters each.
        total += format_then_measure(10)
        call += 1
    }
    assert_eq(total, 30 * CALLS)
}

pub fn test_an_fstring_reaches_a_predicate() -> void {
    var seen = 0
    var call = 0
    while call < CALLS {
        seen += format_then_compare(10)
        call += 1
    }
    assert_eq(seen, 10 * CALLS)
}
