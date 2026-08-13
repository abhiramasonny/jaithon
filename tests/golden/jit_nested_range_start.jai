#: What an INNER `for … in a..b` may assume about the range it iterates.
#:
#: The head of a compiled loop keeps its counter in a register (see
#: jit_osr_range_bias). A loop nested inside a compiled body does not: it builds
#: a real ObjIter and an ObjRange, and used to re-derive the yielded value as
#: `start + index * step` from three loads of them on every single iteration.
#:
#: A range this tier builds itself always steps by 1 -- the helper it calls
#: takes no step -- and when the low end is an integer literal the start is a
#: compile-time constant too, so the value is just `index + K`. Both facts are
#: recorded on the operand-stack entry that holds the iterator, and getting
#: either wrong is a plausible wrong answer rather than a crash: an inner loop
#: that counts from zero when it should count from K sums the wrong pairs and
#: says nothing about it.
#:
#: Each function below is one way that could go wrong. The outer loop is what
#: gets compiled; the inner one is the code under test.
#:
#:   * K = 0, the shape everything else is written against;
#:   * K positive and negative and small enough to be an immediate;
#:   * K too large for an immediate, which has to materialise the constant;
#:   * a start that is NOT a literal, which must keep deriving it from the
#:     range -- and one that is not even loop-invariant;
#:   * an inclusive range, whose length is one greater;
#:   * an empty inner range, which must run zero times rather than once;
#:   * a `break` and a `return` out of the inner loop, which leave the iterator
#:     part-way through;
#:   * an inner loop whose body deoptimises, so the interpreter picks the same
#:     loop up from the ObjIter this code has been writing.

fn zero_start(n: int) -> int {
    var acc = 0
    for _i in 0..n {
        for j in 0..40 { acc += j }
    }
    return acc
}

fn small_positive(n: int) -> int {
    var acc = 0
    for _i in 0..n {
        for j in 7..40 { acc += j }
    }
    return acc
}

fn negative_start(n: int) -> int {
    var acc = 0
    for _i in 0..n {
        for j in -5..40 { acc += j }
    }
    return acc
}

# 5000 does not fit the 12-bit immediate an `add` takes, so the constant has to
# be materialised. It is still small enough to be an OP_INT literal.
fn wide_literal(n: int) -> int {
    var acc = 0
    for _i in 0..n {
        for j in 5000..5040 { acc += j - 5000 }
    }
    return acc
}

# The start is a runtime value, so nothing about it is known at compile time.
fn runtime_start(n: int) -> int {
    var acc = 0
    for i in 0..n {
        for j in i..i + 40 { acc += j - i }
    }
    return acc
}

fn inclusive(n: int) -> int {
    var acc = 0
    for _i in 0..n {
        for j in 3..=40 { acc += j }
    }
    return acc
}

fn empty_inner(n: int) -> int {
    var acc = 0
    for _i in 0..n {
        for j in 40..40 { acc += j + 1 }
    }
    return acc
}

fn broken(n: int) -> int {
    var acc = 0
    for _i in 0..n {
        for j in 6..400 {
            if j == 11 { break }
            acc += j
        }
    }
    return acc
}

fn returned(n: int) -> int {
    var acc = 0
    for _i in 0..n {
        for j in 6..400 {
            acc += j
            if acc > 100000 { return acc }
        }
    }
    return acc
}

# An inner body with a guard in it. Under JAITHON_JIT_DEOPT_STRESS every one of
# those guards fails, so the interpreter picks this inner loop up from the very
# ObjIter the compiled form has been writing -- which is what says the index
# left behind is the zero-based one the interpreter expects and not the biased
# value the fast path yields.
fn guarded_inner(n: int) -> int {
    var xs = []
    for i in 0..40 { xs.push(i * 2) }
    var acc = 0
    for _i in 0..n {
        for j in 4..40 { acc += xs[j] }
    }
    return acc
}

fn main() -> int {
    let n = 30000
    print(zero_start(n))
    print(small_positive(n))
    print(negative_start(n))
    print(wide_literal(n))
    print(runtime_start(n))
    print(inclusive(n))
    print(empty_inner(n))
    print(broken(n))
    print(returned(n))
    print(guarded_inner(n))
    return 0
}
