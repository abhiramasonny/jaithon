# `lib/` — the Jaithon side of Jaithon

Two trees, and the distinction between them is the whole reason this directory
is interesting.

    jaithon/    the compiler, written in Jaithon      84 files, ~31.7k lines
    std/        the standard library                  75 files, ~35.1k lines

`std/` is an ordinary library: it is compiled from source on first use and
cached in `__jaicache__`. `jaithon/` is the compiler itself, and it has to be
able to compile *before there is a compiler* — which shapes everything below.

## The bootstrap window

The binary carries a pre-compiled image of the front end in `boot/seed.bin`.
`warmFrontEnd` (`src/runtime/modules/module.c`) loads the compiler's own closure
with `sLoadingFrontEnd` set, and **only inside that window is the seed
consulted**. Everything the compiler needs *after* it is running — the rest of
`std/`, and anything under `packages/` — is compiled from source on demand.

`boot/seed.manifest` lists the 64 modules in that window. Three consequences,
each of which has cost a broken tree at least once:

* **Editing a seeded file does nothing until `make reseed`.** The binary keeps
  serving the old image. A change that appears to have no effect is usually
  this.
* **A seeded module may only import seeded modules.** Otherwise the tree cannot
  compile *at all*, and since `make reseed` needs a working compiler to build
  the next seed, it cannot rebuild its way out — `boot/` has to be restored from
  a commit. `scripts/gate/seed_closure_check.py` is thirty lines of text and
  catches this before the first reseed.
* **A change to the *emitter* needs two reseeds**, and a third to prove the
  fixpoint: the first reseed is built by the old emitter, the second by the new
  one, and only if the third is byte-identical is the compiler a fixed point of
  itself. `make reseed` re-checks this rather than trusting the compiler that
  just wrote the seed.

The seed is not portable between trees: it is tied to `JAI_BUILD_ID`. Never copy
`boot/` from a worktree — copy the source, then reseed locally.

## The compiler, stage by stage

    lexer.jai  token.jai        source to tokens
    parse/                      tokens to a `jaithon.ast` tree
    resolve.jai  symbol.jai     names to slots, scopes, captures, imports
    check/                      types, in side tables keyed by `id(node)`
    opt/                        bytecode-level optimisation
    emit/                       tree to a bytecode chunk
    jaic.jai                    the driver that runs all of the above, and caches
    diag.jai                    diagnostics, shared by every stage
    repl.jai                    the incremental front end
    mod.jai                     module loading and resolution
    ast.jai  ast/               the tree itself
    ast_encode.jai              the on-disk form
    ast_unparse.jai             the formatter's back end (`jaithon fmt`)
    ast_visitor.jai             the generic walk
    tool/                       `jaithon` subcommands written in Jaithon

## Directories behind a facade

`parse/`, `check/`, `emit/`, `ast/` and `opt/` follow one pattern: the original
file name survives as a **thin facade** that imports the directory and
re-exports exactly what it exported before, so no importer anywhere had to
change. `parser.jai` is 13 lines over 11 modules; `ast.jai` is 97 over 7.

Two idioms make those splits possible, and both are worth copying rather than
reinventing:

* **State class plus free functions.** `check/` is the original: `Ctx` holds the
  state and every checking routine is a free function taking it. A pass split
  this way has no cycles to break as long as the files are ordered so that each
  calls only what is below it.
* **Module-level hooks behind trampolines**, for the cycles that ordering cannot
  remove. `parse/state.jai` declares `var _hook_parse_expression` with a default
  that throws its own name, plus a one-line `parse_expression(p)` that calls
  through it; `parse/wire.jai` installs the real functions at module scope. Code
  holding a `Parser` directly — `repl.jai`, `mod.jai` — keeps working untouched,
  because importing anything from the facade runs the wiring.

## Working here

    make reseed          after editing anything in boot/seed.manifest
    make seed-check      can the seed still bootstrap from a cold cache
    make fixpoint-check  does the compiler compile itself to itself
    ./jaithon fmt --check lib/
    make test

A miscompile poisons `__jaicache__`, and the poisoned bytecode outlives the fix.
After any bad build:

    find lib packages -name __jaicache__ -type d -exec rm -rf {} +
