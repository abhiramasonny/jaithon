# Scripts

Grouped by job, not by language.

`gate/` is what proves the tree is correct: everything `make test` and `make
check` invoke, plus the standalone checks that compare a generated table or
image against a golden. `bench/` is measurement. `dev/` is everything else —
diagnostics you run by hand, and the build helpers that generate or sync
something. `run_tests.sh` and `install.sh` stay at the top because they are the
two commands a person types.

Shell (`.sh`) orchestrates a build or another process — invoking `make`, the
compiled `jaithon` binary, or another script, and checking exit status. Python
(`.py`) parses or checks text and tables — source, disassembly, generated
tables — where Python's string and data handling beat shell. The two `.jai`
files are jai-native steps: they run under the compiler itself, not around it.

Every script resolves the repo root from its own location, so they work from
any working directory. A script moved between these directories has to have
that resolution adjusted by one level.

```text
run_tests.sh            the whole gate (make test runs this)
install.sh              install jaithon to /usr/local

gate/branch_table_check.py   VM verifier's and optimiser's branch tables must agree
                         (make branch-table-check)
gate/check_packages.py       validate workspace package manifests and their deps
                         (make package-check)
gate/fixpoint_check.sh       compile each source twice, diff the images byte for
                         byte (make fixpoint-check)
dev/gen_seed.py             generate boot/seed.c from the compiler's .jaic images
                         (make reseed)
install.sh              build jaithon and install it to a prefix (see
                         README.md; run directly, no make target)
gate/jit_compile_check.py    confirm test_jit_* tests actually reach compiled/OSR
                         code (make jit-compile-check)
dev/jit_declines.sh         capture JIT decline reasons, check against a baseline
                         (make jit-declines-check)
gate/jit_fusion_check.py     fail when an opcode is missing an arm in the function
                         JIT (make jit-fusion-check)
gate/jit_split_check.sh      check for split-operand-bank bugs in JIT compilation
                         (make jit-split-check)
gate/linetable_golden.sh     confirm the line table encodes identically across a
                         change (make linetable-check)
gate/opcode_table_check.py   fail when the VM, front end, and spec opcode tables
                         disagree (make opcode-check)
gate/opstats_check.sh        compare opcode-count histograms across benchmarks
                         (make opstats-check)
dev/render_logo.py          render every shipped logo raster from
                         assets/logo/jaithon.svg -- no make target, run
                         directly (see the script's own header)
bench/run_bench.sh            benchmark jaithon against equivalent CPython programs
                         (make bench)
run_tests.sh            the whole test driver behind `make test`; see
                         tests/README.md for the layers it runs
gate/seed_check.sh           confirm the seed can bootstrap the compiler from a
                         cold cache (make seed-check)
gate/seed_touch.jai          the one .jai file: compiles every module the seed
                         carries, for `make reseed` to embed
dev/stage0_reseed.sh        reseed from a HEAD snapshot of lib/jaithon/compile
                         instead of the working compiler -- no make target,
                         run directly (see CONTRIBUTING.md)
```
