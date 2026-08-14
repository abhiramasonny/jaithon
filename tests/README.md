# Tests

`scripts/run_tests.sh` — the driver behind `make test` — runs six layers in
order; its numbered section banners are the source of truth this file
summarises. `tests/bench` and `tests/fuzz` sit outside that driver and are
run by their own `make` targets.

```text
golden    tests/golden      layer 2 of run_tests.sh: each .jai runs and its
                             stdout is diffed against a checked-in .expected
                             file, at every -O level and under gc/deopt stress
lang      tests/lang        layer 3: unit tests, run via `jaithon test`
                             alongside tests/stdlib and the package tests
stdlib    tests/stdlib      layer 3, same as tests/lang: unit tests run via
                             `jaithon test`; standard-library coverage only
checker   tests/checker     layer 3, same as tests/stdlib: unit tests for the
                             self-hosted compiler's checker/optimiser internals
                             (jaithon.compile.check / opt.chunk), split out of
                             tests/stdlib since they don't test std.* at all
errors    tests/errors      layer 4: each .jai is compiled with `jaithon
                             check` and must produce the diagnostic named in
                             its `# expect: Exxxx` header
repl      tests/repl        layer 5: .repl transcripts are fed to `jaithon
                             repl`, checked against .expected (and optional
                             .expected-err / .expected-exit) files
bench     tests/bench       not part of run_tests.sh: run by `make bench`
                             (scripts/run_bench.sh), which checks each .jai
                             against an equivalent-output CPython port before
                             timing it; see tests/bench/README.md
fuzz      tests/fuzz        not part of run_tests.sh: run by `make kind-fuzz`
                             (tests/fuzz/kind_mutation.py and
                             iter_mutation.py), excluded from `make test`
                             because it takes several minutes
vm        tests/vm          layer 1 (the bytecode verifier) plus C binaries
                             built by dedicated Makefile targets -- see below
```

Layer 1, the verifier, is also where `tests/vm` gets exercised: `run_tests.sh`
invokes the `verify_chunk` binary and the `tests/vm/*.sh` scripts
(`field_kind_disasm.sh`, `sidecar.sh`, `cache_corrupt.sh`) directly. The rest
of `tests/vm/*.c` (`crc32_equiv.c`, `chunk_caches.c`, `linetable_ltv1.c`,
`jit_arena.c`, `jit_arm64.c`, `field_natives.c`, `invoke_result_kind.c`) is
built by its own named Makefile target (e.g. `make jit-test`) and not run
from `run_tests.sh` at all. **`tests/vm` therefore mixes two things with no
naming cue to tell them apart:** Makefile-built C binaries (`tests/vm/*.c`)
and shell-driven checks (`tests/vm/*.sh`, invoked from
`scripts/run_tests.sh`) sit in the same directory, and only reading the
Makefile or `run_tests.sh` tells you which is which.

Formatting (layer 6, `jaithon fmt --check`) and the run summary (layer 7)
close out `run_tests.sh` but check no `tests/` subdirectory of their own.
