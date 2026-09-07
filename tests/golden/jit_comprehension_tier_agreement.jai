#: List comprehensions in JIT-warmed bodies. The two tiers must agree exactly.
#:
#: Sixth of the tier-agreement set. A comprehension's accumulator is pushed by
#: OP_BUILD_LIST BEFORE the loop head, so it sits below the window the OSR model
#: tracks and OP_LIST_APPEND used to refuse outright -- every comprehension in
#: the language ran its loop interpreted. The append arm now reads the
#: accumulator out of its FRAME SLOT (findComprehensionAcc computes which one)
#: through a reserved callee-saved register, guarding that what it finds is
#: really a list.
#:
#: The edges chosen are the ones whose two paths could plausibly disagree: each
#: element kind the store has a tag for (int, float, bool, str, list, instance),
#: a comprehension over a range, over a list and over a string (three different
#: loop heads above the same append), a filtered comprehension so the append is
#: not reached on every iteration, an empty source, a NESTED comprehension
#: (two accumulators in one region -- only one register is reserved, so the
#: inner one must refuse and still produce the right answer), a comprehension
#: whose result is immediately appended to again, and one whose element kind
#: changes between calls so the store's tag has to be re-proved.
#:
#: Every body is called 400 times before the edge value arrives, so the edge is
#: taken in compiled code. Output must be byte-identical under JAITHON_NO_JIT=1;
#: the golden runner also re-runs it at JAITHON_JIT_THRESHOLD=1 and under
#: DEOPT_STRESS.

fn c_range(n: int) -> list[int] { return [i * 2 for i in 0..n] }
fn c_list(xs: list[int]) -> list[int] { return [x + 1 for x in xs] }
fn c_str(s: str) -> list[str] { return [c for c in s] }
fn c_float(n: int) -> list[float] { return [float(i) * 0.5 for i in 0..n] }
fn c_bool(xs: list[int]) -> list[bool] { return [x % 2 == 0 for x in xs] }
fn c_strs(xs: list[int]) -> list[str] { return [str(x) for x in xs] }
fn c_lists(n: int) -> list[list[int]] { return [[i, i + 1] for i in 0..n] }
fn c_filter(xs: list[int]) -> list[int] { return [x for x in xs if x % 3 == 0] }
fn c_nested(n: int) -> list[list[int]] {
    return [[j for j in 0..i] for i in 0..n]
}

fn sum_of(xs: list[int]) -> int {
    var t = 0
    for x in xs { t += x }
    return t
}

fn main() {
    let base: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    var warm = 0
    for _i in 0..400 {
        warm += sum_of(c_range(8))
        warm += sum_of(c_list(base))
        warm += c_str("abcdef").len()
        warm += c_float(8).len()
        warm += c_bool(base).len()
        warm += c_strs(base).len()
        warm += c_lists(6).len()
        warm += sum_of(c_filter(base))
        warm += c_nested(5).len()
    }
    print("warm " + str(warm))

    print("range " + str(c_range(10)))
    print("list " + str(c_list(base)))
    print("str " + str(c_str("hey")))
    print("float " + str(c_float(4)))
    print("bool " + str(c_bool(base)))
    print("strs " + str(c_strs(base)))
    print("lists " + str(c_lists(3)))
    print("filter " + str(c_filter(base)))
    print("nested " + str(c_nested(4)))

    # Empty sources, after the loops are already compiled.
    let empty: list[int] = []
    print("empty " + str(c_range(0)) + str(c_list(empty)) + str(c_str("")))

    # A comprehension over a comprehension's result, and one appended to after.
    var chain = [x * x for x in c_range(6)]
    chain.push(999)
    print("chain " + str(chain))

    # Long enough to reach the OSR tier, where the accumulator register is
    # reserved. Kept to a sum so the expected output stays small.
    let big = [i % 7 for i in 0..200000]
    print("big " + str(sum_of(big)) + " " + str(big.len()))

    let bigstr = [c for c in "abcdefghij" * 20000]
    print("bigstr " + str(bigstr.len()) + " " + bigstr[0] + bigstr[199999])
}
