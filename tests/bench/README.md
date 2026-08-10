# Benchmarks

Each benchmark is a program written the same way in every language it has a
port for. The `.jai` file is the reference; a peer only counts if it prints the
same output, which the runner checks on every run — a benchmark that computes
the wrong thing quickly is not a benchmark.

    tests/bench/loop_sum.jai        required
    tests/bench/loop_sum.py         python3 column
    tests/bench/loop_sum.cpp        c++ column      (optional)
    tests/bench/LoopSum.java        java column     (optional, CamelCase name)

Run with `make bench`, or `RUNS=3 scripts/run_bench.sh` for a quicker pass.
Columns whose toolchain is missing disappear; benchmarks without a port show
`—`. C++ and Java are compiled once outside the timed runs, because the table is
about how fast the program runs and not how fast it builds.

## What the C++ and Java columns are for

They are a **scale, not a target**. CPython alone cannot tell you whether a gap
is worth chasing or already near the floor. C++ at `-O2` is roughly what the
machine can do with the program; Java is what a mature JIT does with the same
source. Together they bracket the interpreter.

Read them that way. From one run on an M2 Max:

| | jaithon | python3 | c++ | java |
|---|---|---|---|---|
| loop_sum | 643ms | 2279ms | 36ms | 73ms |
| alloc_churn | 85ms | 158ms | 4ms | 48ms |

`loop_sum` says the interpreter is 3.5x CPython and 18x off C++ — and that the
distance to Java, 8.8x, is not something instruction dispatch alone closes.
That is the honest shape of the problem: beating CPython by a few times is
interpreter work, and approaching Java is a JIT.

`alloc_churn` says something different. The gap to Java is 1.8x, far tighter
than `loop_sum`'s, because both are paying for a collector rather than for
dispatch. A benchmark whose Java column is close is one where the remaining win
is small; a benchmark whose Java column is far away has headroom.

## Adding one

Write the `.jai` first and make it print something derived from the whole
computation, so a wrong answer cannot look fast. Add the `.py` and check both
by hand before trusting the table. C++ and Java ports are optional and worth
adding for anything you intend to optimise, since without them you cannot tell
a real gap from an already-tight one.

Keep the runtime in the same range as the others — roughly 50–200ms for
jaithon, with `loop_sum` the deliberate outlier — so no single benchmark
dominates the total.
