# Contributing to Jaithon

## Where things live

| Stage | Files |
|---|---|
| Lexing | `src/lang/lexer.c` `token.c` |
| Parsing | `src/lang/parser.c` (plumbing, entry points) `parser_expr.c` `parser_stmt.c` `parser_decl.c` `parser_type.c`, sharing `parser_internal.h` |
| AST | `src/lang/ast.c` |
| Resolving | `src/sema/resolve.c` (names, slots, upvalues) `modsig.c` |
| Checking | `src/sema/check.c` (plumbing, entry points) `check_expr.c` `check_stmt.c` `check_decl.c` `check_fold.c`, sharing `check_internal.h`; the type lattice is `types.c` |
| Lowering | `src/codegen/codegen.c` (the emitter's shared machinery and the entry points) `codegen_expr.c` `codegen_stmt.c` `codegen_decl.c` `codegen_pattern.c`, sharing `codegen_internal.h` |
| Optimising, verifying | `src/codegen/optimize.c` |
| Executing | `src/vm/vm.c` — one file on purpose; see below |
| Values, objects, GC | `src/vm/value.c` `object.c` `table.c` `chunk.c` `gc.c` |
| `.jaic` images | `src/vm/serialize.c` |
| Builtins, modules | `src/runtime/` |
| Threads, process, Metal | `src/native/` |
| CLI and REPL | `src/cli/` |

The `*_internal.h` headers are the interface between the files of one stage and
nothing else: no file outside a stage's own directory may include one.

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

## Verify bootstrap

```bash
make bootstrap
```

This compiles every file in `lib/` and `tests/` with both the C front end and
the self-hosted one in `lib/jaithon/compile/`, then diffs the serialised
bytecode. Any divergence prints the first differing instruction.

**Both halves of the back end are mirrored, so both carry the obligation.**

| you change | mirror it in |
|---|---|
| lowering in `src/codegen/codegen*.c` | `lib/jaithon/compile/emit.jai` |
| a pass in `src/codegen/optimize.c` | the matching file in `lib/jaithon/compile/opt/` |
| the pass order in `optimizeFunction` | `lib/jaithon/compile/opt/mod.jai` |
| the decode/rebuild machinery (`codeDecode`, `markUnreachable`, `codeRebuild`) | `lib/jaithon/compile/opt/chunk.jai` |

The pass order is not an implementation detail: the two front ends are compared
instruction for instruction, so it is part of the specification. So is anything
that decides *when* a constant enters the pool — `emitAssign` interns an
assigned field's name before it emits the object expression, and an emitter that
interns it later produces the same instructions with different operands.

`make bootstrap` does not currently report zero, and the reason is worth knowing
before you read its output. The self-hosted front end has no type checker
(ROADMAP.md, P0), so a `const` is inlined on one side and loaded on the other,
an `any`→`T` boundary picks up an `OP_TYPE_GUARD` on one side only, and the cast
wrapper the checker leaves behind stops the C from tail-calling a returned call.
Those land *first* in almost every file and the report stops at the first
divergence, so they hide whatever the back end did. Two things to do about it:

- **Compare at `-O0` as well.** `./jaithon --bootstrap-verify -O0 lib tests`
  runs both sides at the level you name. A file that differs at `-O2` and agrees
  at `-O0` is the optimiser's (or `jaiChkTryFold`'s, which is gated on
  `optLevel > 0`); a file that differs at both is the emitter's or the checker's.
- **Use the two files that reach the back end.**
  `tests/golden/optimiser_bootstrap.jai` and
  `tests/golden/emitter_bootstrap.jai` annotate every binding `any` and declare
  no `const` and no return types, so the checker rewrites nothing in them. They
  read `ok` today and a `DIFF` on either one is yours. Add to them when you add
  a construct or a pass rule; nothing else in the tree can see a back-end
  divergence.

Before concluding anything from a bootstrap run, clear the caches:
`find . -name __jaicache__ -type d -exec rm -rf {} +`. A stale one has produced
several false diagnoses.

## Formating

```bash
jaithon fmt .
jaithon fmt --check .    # CI gate; non-zero if anything is unformatted
```

## Adding a language feature

1. Write the specification change first, in `spec/LANGUAGE.md`.
2. Add a golden test showing it working and an error test showing it failing.
3. Lexer --> parser --> AST --> resolver --> checker --> codegen --> VM, in that order. Each stage should compile and its own tests pass before you start the next.
4. Mirror the front-end changes in `lib/jaithon/compile/` and run `make bootstrap`.
5. Update `editors/vscode/syntaxes/jaithon.tmLanguage.json` if it introduces
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
