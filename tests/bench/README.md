# Benchmarks

Each benchmark is a program written the same way in every language it has a
port for. The `.jai` file is the reference; a peer only counts if it prints the
same output, which the runner checks on every run — a benchmark that computes
the wrong thing quickly is not a benchmark.

Each benchmark lives in its own subdirectory, named for the benchmark:

    tests/bench/loop_sum/loop_sum.jai        required
    tests/bench/loop_sum/loop_sum.py         python3 column
    tests/bench/loop_sum/loop_sum.cpp        c++ column      (optional)
    tests/bench/loop_sum/LoopSum.java        java column     (optional, CamelCase name)

Run with `make bench`, or `RUNS=3 scripts/run_bench.sh` for a quicker pass.
GPU/ML training benches live under `tests/bench/jaitensor/` and are **not**
part of that table — run them with `make bench jaitensor`. The jaitensor
suite compares jaitensor against a 1:1 PyTorch MPS copy on fifteen MLP
workloads plus per-shape GEMM and attention microbenches. Each program warms
its GPU graph, records only the synchronized device-work interval, and emits
the exact work it executed. The table reports the **median of 5** and the
coefficient of variation (override with `RUNS=`), rather than selecting the
fastest process. MLP GFLOPS is `6ND` over full batches actually processed;
GEMM is `2MNK` per multiply and attention counts QK plus PV. The `route` column
shows Jaithon's selected GEMM family.

Every timed run also checks values derived from its GPU output. GEMM checks
finite, bounded, nonzero result samples; attention additionally checks the
softmax convex bound; MLP requires finite positive loss that does not diverge.
`result: ok` therefore describes computed values, not matching metadata.

The PyTorch GEMM peer intentionally avoids `out=` inside autocast: PyTorch
implements `out=` matmul as float32, bypassing autocast. It also disables the
cross-call autocast cache because Jaithon's float32 buffers are recast on each
dispatch (as changing activations and weights are during training). Both peers
therefore expose the same float32-storage, float16-compute, float32-result
contract.
Set `JAITENSOR_PYTHON` to a PyTorch install with MPS (defaults to
`/tmp/jaitensor-bench/bin/python` when present). Use
`LEVEL=easy|medium|hard` to shorten epochs/repeats.
`tests/bench/jaitensor/gpu_util.py` is a separate ioreg sampler for
utilization JSON — not part of `make bench jaitensor`. Columns whose
toolchain is missing disappear; benchmarks without a port show
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

## The holdout set

Five of these — `poly_dispatch`, `json_parse`, `heat_2d`, `graph_bfs`,
`error_paths` — are a **holdout**. They were written after the optimisation
work on the other twenty was finished, deliberately covering shapes that work
did not target: a call site with eight receiver classes, a hand-written parser,
a 2D float stencil, pointer-chasing search, and exception paths.

They are not more important than the other twenty. They are the check that the
other twenty are not being fitted. When a change makes one of the original
twenty faster and leaves the holdouts alone, that is the evidence it
generalised — and when it moves a holdout backwards, that is a regression the
original twenty cannot see.

They found something the day they landed: `json_parse` and `error_paths` are
the same speed with `JAITHON_NO_JIT=1` as without it, because every function in
both declines from the compiled tier. See `docs/roadmap.md` §8.

## Adding one

Make a subdirectory named for the benchmark and write the `.jai` there first,
making it print something derived from the whole computation, so a wrong
answer cannot look fast. Add the `.py` alongside it and check both by hand
before trusting the table. C++ and Java ports are optional and worth adding
for anything you intend to optimise, since without them you cannot tell a real
gap from an already-tight one.

Keep the runtime in the same range as the others — roughly 50–200ms for
jaithon, with `loop_sum` the deliberate outlier — so no single benchmark
dominates the total.
