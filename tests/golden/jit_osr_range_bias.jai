#: What a range-headed compiled loop keeps in registers, and what it derives.
#:
#: A range head used to reserve four callee-saved registers: the slots pointer,
#: the iterator's index, its limit, and the range's START, which was added to
#: the index once per iteration to make the yielded value. The start is now
#: folded into the index and the limit in the prologue, so the loop variable IS
#: the index register, the add is gone from the loop, and every range-headed
#: loop in the language has one more register for its locals.
#:
#: That is only sound while the biased values mean what the unbiased ones did,
#: and each case below is one of the ways they might not:
#:
#:   * a start below zero, and a start large enough that the bias moves the
#:     index across most of the type;
#:   * a `break`, which leaves the loop with the iterator part-way through --
#:     ObjIter::index is zero-based, so every way out has to take the start back
#:     off before writing it, and the interpreter has to carry on from the right
#:     element;
#:   * a range spanning the whole of int64, whose length saturates at INT64_MAX
#:     so that `start + limit` cannot be represented at all. jaiJitEnterOsr
#:     refuses to enter a compiled form for one of those, and the answer has to
#:     come out the same anyway;
#:   * an inner range that starts at the outer variable, which is where a
#:     nonzero start turned up in the first place -- `for j in i + 1..n` is
#:     nbody's advance -- and a nested pair re-enters the head repeatedly.
fn negstart() -> int {
    var acc = 0
    for i in -1000..200000 { acc += i }
    return acc
}

fn bigstart() -> int {
    var acc = 0
    for i in 4611686018427387000..4611686018427587000 { acc = acc + i % 7 }
    return acc
}

fn breaker() -> int {
    var seen = 0
    var last = 0
    for i in 5..500000 {
        if i == 400000 { break }
        seen += 1
        last = i
    }
    return seen * 1000000 + last
}

fn saturated() -> int {
    var acc = 0
    for i in -9223372036854775807..9223372036854775807 {
        acc += 1
        if acc > 300000 { break }
    }
    return acc
}

fn nested() -> int {
    var acc = 0
    for i in 3..900 {
        for j in i + 1..900 { acc += j - i }
    }
    return acc
}

fn main() -> int {
    print(negstart())
    print(bigstart())
    print(breaker())
    print(saturated())
    print(nested())
    return 0
}
