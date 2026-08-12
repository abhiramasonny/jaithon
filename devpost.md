# Jaithon 3

## Inspiration

I wrote the first Jaithon in June 2023. It was 1,371 lines of C in a single file, `src/interpreter.c`, and I built it to teach kids (my brother) how to program. It was incredibly simple, with the syntax looking like the syntax below.

```
var n1 = 0
var n2 = 1
input nterms
while count < nterms do
    print n1
loop
```

It evaluated line by line as it parsed, so every feature I added made the next one harder to add, and that one file larger and larger. Jaithon 1 was primarily made to teach myself how C code works, and memory management, when I was in 8th grade.

Jaithon 2 shipped in December 2025 with a bytecode compiler, a Cocoa GUI module, and Metal compute. It was 12,327 lines of C and 13,342 lines of standard library code (bootstrapped in jaithon itself). Technically speaking, it worked, however the general archetecture was still wrong. The syntax of the language was also pretty bad, types got checked at the moment they were used, so a type error surfaced inside code generation. Names were resolved during codegen aswell, which meant a scoping bug printed as a bytecode bug. The bytecode optimizer was also terribly written.

On August 5, 2026 I deleted all of it, and decided to build Jaithon 3. Jaithon 3 is 59,944 lines of C, plus 55,673 lines of Jaithon 3, with the PERFECT syntax, and also incredibly efficient and fast.

## What it does

Jaithon is a garbage-collected language with a bytecode VM. The structure of the language is inspired from Java, with declared fields, explicit `pub` visibility, traits with default methods, type annotations that get enforced, etc. Most of the syntax comes from a mix of Python and JS, with no semicolons, f-strings, comprehensions, first-class functions, var let and const for variable declarations, etc.

Here is some example Jaithon 3 Code:

```Python
enum Shape {
    Circle(radius: float),
    Rect(w: float, h: float),
}

fn area(s: Shape) -> float {
    return match s {
        Shape.Circle(r)  => math.PI * r ** 2,
        Shape.Rect(w, h) => w * h,
    }
}

let squares = [x ** 2 for x in 0..10 if x % 2 == 0]
let first_ten = iter(source).map(parse).filter(is_valid).take(10).collect()
```

The Jaithon binary does everything you would need to do: `run`, `repl`, `check`, `build`, `fmt`, `test`, `doc`, `bench`, `disasm`. Diagnostics carry rustc-style codes, with the error codes being intentionally designed to be incredibly easy to debug. Errors are from `E00xx` through `E08xx`, and are grouped by the stage that raised them.

On my M2 Max Macbook Pro against CPython 3.13.13:

| Program | Jaithon | Python | Ratio |
|---|---|---|---|
| `loop_sum` ($5 \times 10^7$ iterations) | 1.08 s | 3.10 s | $2.9\times$ |
| `object_dispatch` | 0.17 s | 0.30 s | $1.8\times$ |
| `dict_ops` | 0.06 s | 0.13 s | $2.2\times$ |
| `list_ops` | 0.09 s | 0.17 s | $1.9\times$ |
| `fib(30)` | 0.12 s | 0.14 s | $1.2\times$ |
| `string_build` | 0.05 s | 0.05 s | tie |

Naive `fib(30)` makes $2F_{31} - 1 = 2{,}692{,}537$ calls, so 0.12 s of wall clock works out to roughly $2.2 \times 10^7$ calls per second including process startup.

## How I built it

There were 6 stages to building Jaithon 3.

```
source → lexer → parser → resolver → type checker → codegen → VM
           |        |         |            |            |        |
        tokens     AST    symbols +     types +     bytecode  values
                          slots         casts      + caches   + GC
```

The resolver assigns every name a slot in the stack, and settles a majority of the capture semantics before any type is assigned / examined. The type checker then can run on a fully resolved tree, which speeds up builds, and also makes sure that it doesnt have to guess what type a name is. Codegen gets the finished tree and emits bytecode against my custom 107-opcode instruction set for Jaithon 3, which then goes into the VM to be run.

Property access goes through polymorphic inline caches keyed on shape IDs. A site starts monomorphic, grows to polymorphic when it sees a second shape, and bails to the generic path once it goes megamorphic.

Jaithon is also extremely bootstrapped, with everything above the raw primitives being written in jaithon itself, including the REPL. `fmt`, `test`, `doc`, and `bench` are all Jaithon programs which `main.c` dispatches to. The standard library covers collections, graphs, matrices, big integers, regex, JSON, threads, and sorting. `packages/jaiplot` is a plotting library which I developed in pure Jaithon with SVG, PNG, and live-animation on windows at 120 FPS. `lib/jaithon/compile` is a second lexer, parser, resolver, and emitter, also in Jaithon, which exists so in the future, the frontend can be written ENTIRELY in Jaithon, and only the bytecode VM is written in C. This would allow for complete modularity and allow Jaithon to be incredibly customized and optimized for the future. At the current moment, this second compiler works fine, however isnt as optimized as the C compiler, so this will be a thing for a future update.

## Challenges I ran into

### Problem 1

Every optimizer pass works on a decoded instruction list where branch operands point at instruction *indices*, which is something that carried over from Jaithon 2. The hard part is re-encoding. Let $s_k(i)$ be the encoded size of instruction $i$ at iteration $k$, and let $L_k(i) = \sum_{j < i} s_k(j)$ be its byte offset. A branch fits in `i16` only when

$$|L_k(t) - L_k(i)| < 2^{15}$$

Shrinking one branch shifts every offset that follows, which can make another branch shrink too. I therefore iterate $L_{k+1} = \mathrm{encode}(L_k)$ until $L_{k+1} = L_k$. Sizes never grow between iterations, so the sequence is monotone, bounded below, and the loop terminates. If a branch still does not fit after the fixed point settles, the pass abandons its work and returns the chunk untouched. I would rather miss an optimization than emit a wrong one.

### Problem 2

That same re-encode also has to rebuild the line table, the exception table, and the default-thunk offsets against the new layout. Missing any of those would send tracebacks to the wrong line, or make a `try` region cover code it never covered before.

### Problem 3

`jaiVerifyChunk` runs on every function codegen emits. It checks operand widths and ranges, branch targets, exception handlers, default-thunk entry points, and stack depth consistency. Debug builds also verify on both sides of the optimizer and panic if a chunk arrived valid and left broken.

## What I learned

Splitting the front end into stages that each finish before the next starts was worth the time that it took. Jaithon 2 put name resolution and type checking into an 894-line `compiler.c` program which frankly was unmanagable. As a result of this, a scoping bug there surfaced as a bad opcode and many many other errors and bugs that I probably didnt notice were in play. Jaithon 3 spent 9,397 lines on `src/sema` alone, and now the error code tells me which stage broke before I open a file. Ensuring that a majority of code is written in Jaithon itself also allows me to catch errors with the bytecode VM while I was writing the standard library and jaiplot. It also allowed me to verify that Jaithon was working as intended, and the modularization was definatly worth it.

Writing the formatter, test runner, doc generator, and benchmark harness in Jaithon exposed more design mistakes than the test suite did. Something I learnt is that code that actually was running in jaithon that helped it to compile is much much harder to build than I originally anticipated. This is one of the reasons bootstrapping the frontend is taking so long, I need it to be accurate firstmost, and I also need it to be fast.

## What's next

Close the 99 bootstrap divergences so the self-hosted front end emits bytecode identical to the C one, so that its in a point where I can optimize it even more.

I also want to make Jaithon MUCH more fast, currently it is roughly 1.5x faster than CPython, I want to get it to a point where it rivals C++ for speed, and can be used to write real applications in it. Currently Jaithon exists just as the perfect syntax that I scrapped together from various programming languages (Java, Python, C++, NodeJS, Bash, Lua, Rust) in (my opinion) the perfect architecture (Javas), however in the future I would want to expand Jaithon to be targeted towards a specific field. Machine Learning definatly has some possibility, as currently nearly all machine learning work is done in Python. Python however isnt the optimal language to do machine learning in, as CPython isnt as optimized as possible. Even though training runs in C, it could be much faster which Jaithon definatly has the potential to get to.
