# src

The C core. Everything here is the part of Jaithon that cannot be written in
Jaithon: the interpreter, the collector, the compiled tier, and the platform.
The front end — lexer, parser, resolver, type checker, emitter, optimiser — is
not here. It is written in Jaithon, lives in [`lib/jaithon/`](../lib/jaithon/),
and is loaded from [`boot/seed.bin`](../boot/) at startup.

## Layout

```text
common/       diagnostics and the allocator
vm/           values, the interpreter loop, the collector, the hash table
  object/     the heap object types
  bytecode/   chunks, the .jaic container, the verifier
  jit/        the compiled tier
  trace/      @trace instrumentation
runtime/      the builtin surface and the module loader
  builtins/   builtin functions and methods, by family
  modules/    import resolution, the seed, module state
native/       platform code
  apple/      Metal, CoreML, camera, Cocoa
  posix/      processes and threads
cli/          argument parsing, the REPL, the subcommands
```

## Layering

Dependencies run downward through `common` → `vm` → `runtime` → `native` →
`cli`, and an audit of every `#include` between directories found exactly two
edges going the other way. Both are deliberate and documented at the include
site: `vm/vm.c` and `vm/jit/jit_body.c` reach into `runtime/runtime.h` for the
builtin name table, which is what lets `xs.len()` resolve to a native. Nothing
else in `vm/` knows the runtime exists.

`vm/object`, `vm/bytecode` and `vm/` proper are siblings rather than a stack:
`chunk.h` needs `value.h`, `object.c` needs `gc.h` and `table.h`. Treating any
one of them as strictly lower than the others is what makes the graph look
cyclic when it is not.

## Where the size is

`vm/jit/` is the largest directory by a wide margin, and that is expected: a
compiler back end that emits arm64 by hand is simply a lot of code. It is 18
files sharing one internal header rather than the single 17,000-line file it
used to be. `jit.h` is the boundary the rest of the VM sees; `jit_internal.h`
is the shared state the tier's own files pass around, and nothing outside
`vm/jit/` includes it.

## Adding a file

The Makefile finds sources by directory glob, so a new `.c` in an existing
directory is compiled with no Makefile change. The globs are

```text
src/common/*.c
src/vm/*.c            src/vm/*/*.c
src/runtime/*.c       src/runtime/*/*.c     src/runtime/*/*/*.c
src/native/*.c        src/native/*/*.c      src/native/apple/*.m
src/cli/*.c           src/cli/*/*.c
```

so a directory nested deeper than these is silently not built. `runtime/` is
the only tree that reaches three levels.

Two things a new file in `vm/jit/` has to get right. It must reproduce the
`#if (defined(__aarch64__) || defined(__arm64__))` guard the tier is wrapped
in, and the non-arm64 fallback stubs must continue to exist exactly once
across the whole directory. Neither mistake fails the build on this machine.
