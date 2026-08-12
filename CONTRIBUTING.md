# Contributing to Jaithon

## Where things live

| Area | Files |
|---|---|
| Compiler | `lib/jaithon/compile/` |
| Interpreter | `src/vm/vm.c` |
| Values, tables, GC | `src/vm/value.c`, `table.c`, and `gc.c` |
| Heap objects | `src/vm/object/` |
| Bytecode and `.jaic` images | `src/vm/bytecode/` |
| arm64 JIT | `src/vm/jit/` |
| Builtins | `src/runtime/builtins/` |
| Imports and compiler boot | `src/runtime/modules/` |
| Platform implementations | `src/native/apple/` and `src/native/posix/` |
| CLI and REPL | `src/cli/` |

See `src/README.md` for the full directory map. A private `*_internal.h` header
belongs to the files in its directory; code outside that group must use a
public subsystem header.

`src/vm/vm.c` is the exception to the rule that a 5,000-line file wants
splitting. It was split four ways along its real seams — the call protocol,
unwinding, property access, the public entry points — and measured: with the
interpreter loop alone in its translation unit, `tests/bench/loop_sum` lost
4.3% and `fib_recursive` 7.9%, against a ±1% control (the same objects relinked
in a different order, and a null reordering of two helpers in the file). The
loop is a computed-goto dispatch, its register allocation is one problem across
the whole function, and every call it makes to another translation unit is
opaque to that allocation. Navigability is not worth 3% of the interpreter, so
the file stays whole; the section banners inside it are the index.

## Build

```bash
make            # release, -O2
make debug      # -O0 -g3, assertions on, JAI_DEBUG defined
make clean
```

## Test

```bash
make test                       # everything
scripts/run_tests.sh -v basics  # js one thing
```

Everything must pass before a pull request is merged. You also need to run the suite under GC stress before sending anything that touches the runtime:

```bash
./jaithon --gc-stress test tests/
```

## Verify the compiler seed and fixpoint

```bash
make seed-check
make fixpoint-check PATHS=lib
```

`seed-check` starts the compiler from the embedded `.jaic` seed with source
caches excluded. `fixpoint-check` asks that compiler to compile each selected
source twice and compares the resulting images byte for byte. Run `make reseed`
after changing a module that the compiler needs during startup.

## Formatting

```bash
jaithon fmt .
jaithon fmt --check .    # CI gate; non-zero if anything is unformatted
```

## Adding a language feature

1. Write the specification change first, in `spec/LANGUAGE.md`.
2. Add a golden test showing it working and an error test showing it failing.
3. Lexer --> parser --> AST --> resolver --> checker --> codegen --> VM, in that order. Each stage should compile and its own tests pass before you start the next.
4. Implement compiler changes in `lib/jaithon/compile/` and VM changes under `src/vm/`.
5. Run `make fixpoint-check`; run `make reseed` when the compiler's startup closure changes.
6. Update `editors/vscode/syntaxes/jaithon.tmLanguage.json` if it introduces
   syntax.

## Performance work

```bash
make bench
```

Benchmarks compare against equivalent CPython programs and verify that both
produce identical output before reporting a time — a benchmark that computes
the wrong answer quickly is not a benchmark.

Before optimising, measure. `--stats` reports instruction counts, call counts,
inline-cache hit rates, and GC statistics. `--debug-trace` disassembles every
instruction as it executes.

## Commit style

One logical change per commit. Present tense, imperative:

```
parser: accept trailing commas in argument lists
vm: cache field slots on the shape id, not the class pointer
std.algo: replace the quadratic dedupe in group_by
```

A commit that changes behaviour includes its test in the same commit.
