# Scripts

Three kinds of file live here, by language and job. Shell (`.sh`) orchestrates
a build or another process — invoking `make`, the compiled `jaithon` binary,
or another script, and checking exit status. Python (`.py`) parses or checks
text and tables — source, disassembly, generated tables — where Python's
string and data handling beat shell. The one `.jai` file, `seed_touch.jai`, is
a jai-native bootstrap step: it runs under the compiler itself, not around it.
Most scripts here are wired to a `make` target; `render_logo.py` and
`stage0_reseed.sh` are not, so they are otherwise discoverable only by reading
this directory listing.

```text
branch_table_check.py   VM verifier's and optimiser's branch tables must agree
                         (make branch-table-check)
check_packages.py       validate workspace package manifests and their deps
                         (make package-check)
fixpoint_check.sh       compile each source twice, diff the images byte for
                         byte (make fixpoint-check)
gate/                   checkers that pin a set nobody may quietly add to; see
                         the note below the listing
gate/kind_tag_check.py  pin every place the JIT builds a Value tag out of a
                         compile-time SlotKind (make kind-tag-check)
gen_seed.py             generate boot/seed.c from the compiler's .jaic images
                         (make reseed)
install.sh              build jaithon and install it to a prefix (see
                         README.md; run directly, no make target)
jit_compile_check.py    confirm test_jit_* tests actually reach compiled/OSR
                         code (make jit-compile-check)
jit_declines.sh         capture JIT decline reasons, check against a baseline
                         (make jit-declines-check)
jit_fusion_check.py     fail when an opcode is missing an arm in the function
                         JIT (make jit-fusion-check)
jit_split_check.sh      check for split-operand-bank bugs in JIT compilation
                         (make jit-split-check)
linetable_golden.sh     confirm the line table encodes identically across a
                         change (make linetable-check)
opcode_table_check.py   fail when the VM, front end, and spec opcode tables
                         disagree (make opcode-check)
opstats_check.sh        compare opcode-count histograms across benchmarks
                         (make opstats-check)
render_logo.py          render every shipped logo raster from
                         assets/logo/jaithon.svg -- no make target, run
                         directly (see the script's own header)
run_bench.sh            benchmark jaithon against equivalent CPython programs
                         (make bench)
run_tests.sh            the whole test driver behind `make test`; see
                         tests/README.md for the layers it runs
seed_check.sh           confirm the seed can bootstrap the compiler from a
                         cold cache (make seed-check)
seed_touch.jai          the one .jai file: compiles every module the seed
                         carries, for `make reseed` to embed
stage0_reseed.sh        reseed from a HEAD snapshot of lib/jaithon/compile
                         instead of the working compiler -- no make target,
                         run directly (see CONTRIBUTING.md)
```

## `gate/`

A checker in here enforces a rule that is mechanical but whose *exceptions* are
a judgement, so it pins the current set in a manifest and fails on anything not
in it -- in both directions, since a stale pin guards nothing. Each one takes
`--write` to regenerate the mechanical columns, and each is written so that
regenerating alone cannot make it green: the judgement column has to be filled
in by a person. The script's own docstring says why the rule exists, and what
it cannot see.
