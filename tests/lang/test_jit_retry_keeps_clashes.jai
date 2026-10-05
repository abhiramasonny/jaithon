#: A body whose full compile asks to be retried with one local widened must
#: get that retry, even when a fallback attempt in between stops earlier.
#:
#: `clash_after_match` is built so that slot coalescing hands the dead `b`
#: (a str) to the `match`'s tag (an int): the full walk reaches that second
#: bind and asks for the slot to be compiled dynamic. The `match` arms then
#: get their own retry with the arms off, which stops at the first
#: OP_MATCH_TYPE_POP and never sees the clash -- and it used to OVERWRITE
#: the full attempt's report, so the widening step found nothing to grow and
#: the body was declined every time (JAITHON_JIT_KEEP_RETRY_NEEDS=0 restores
#: that, and the jit-compiles gate then fails).
# jit-compiles: clash_after_match
from std.test import assert_eq

enum Op { Add, Sub, Mul, Div }

let CALLS = 400

fn clash_after_match(o: Op, s: str) -> int {
    let a = s + "!"
    let b = s + "?"
    if a.len() > 100 { return b.len() }
    match o {
        Op.Add | Op.Sub => { return 1 }
        Op.Mul => { return 2 }
        _ => { return 3 }
    }
}

pub fn test_a_body_with_a_clash_after_its_match_compiles_and_answers() -> void {
    let ops = [Op.Add, Op.Sub, Op.Mul, Op.Div]
    let want = [1, 1, 2, 3]
    var call = 0
    while call < CALLS {
        assert_eq(clash_after_match(ops[call % 4], "abc"), want[call % 4], "the arm")
        call += 1
    }
    assert_eq(clash_after_match(Op.Add, "x" * 200), 201, "the early return")
}
