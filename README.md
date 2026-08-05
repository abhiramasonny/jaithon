<div align="center">

<img src="assets/logo/jaithon.svg" alt="Jaithon" width="120">

# Jaithon

**The best parts of Java merged with python**

[Language Guide](LANGUAGE.md)

</div>

## What Jaithon is

Jaithon is a dynamically executed and garbage collected language with a
bytecode VM. It takes heavy insp from the structure from Java (declared fields, forced
visibility, interfaces, type annotations) and insp for everything else
from Python (no semicolons, comprehensions, f-strings, first class functions, and the ease of use).


Pretty much everything (apart from the CORE primitive implementation stuff) is written in jaithon itself, jaithon is VERY much bootstrapped, which makes it easy to develop extra features onto.

---

## Quick start

```bash
git clone https://github.com/abhiramasonny/jaithon
cd jaithon
make                        # builds ./jaithon
make test                   # this is optional, but it runs the benchmarks and tests and stuff
./scripts/install.sh        # also optional, it installs itself to /usr/local
```

The reqs to run jaithon are a C11 compiler and make, readline is used for the REPL if
present. On macOS the Metal and Cocoa frameworks enable the GUI and GPU
modules, however everything else builds and runs without them.

```bash
jaithon run program.jai     # run a file
jaithon                     # REPL
jaithon check src/          # type-check without running
jaithon fmt .               # canonical formatter, no options
jaithon test                # discover and run tests
jaithon doc --out docs/api  # generate API documentation
jaithon disasm program.jai  # bytecode listing
```

The REPL keeps its bindings across lines, continues an unfinished input on a
`... ` prompt, and takes meta-commands. `:help` lists every one of them.

## Examples of .jai Code

```jai
# let is immutable but var is not and const is compile time
let name = "Jaithon"
var count = 0
const MAX = 1 << 16

# types are optional, but they are checked if they are present
let ratio: float = 0.5
let names: list[str] = []
let lookup: dict[str, int] = {}
let maybe: int? = null           # T? is T | null

if names.len() > 0 { print(names[0]) }
print(maybe ?? -1)

# loops and ranges
for i in 0..10 { count += i }
'outer: for row in grid {
    for cell in row {
        if cell == target { break 'outer }
    }
}

# pattern matching
let kind = match code {
    200           => "ok",
    301 | 302     => "redirect",
    400..=499     => "client error",
    n if n >= 500 => "server error",
    _             => "unknown",
}

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

# traits are interfaces with default methods, and they are types.
trait Printable {
    fn to_str(self) -> str
    fn describe(self) -> str { return f"<{self.to_str()}>" }
}

# Errors are classes
fn load(path: str) -> str {
    let file = io.open(path, "r")
    defer { file.close() }
    return file.read()
}

# comphressons and lazy iterators.
let squares = [x ** 2 for x in 0..10 if x % 2 == 0]
let first_ten = iter(source).map(parse).filter(is_valid).take(10).collect()
```

A tour of the whole language is in [`LANGUAGE.md`](LANGUAGE.md).

---

## Errors

Every error is in this format, so hopefully its easy to debug

```
error[E0301]: cannot assign to immutable binding `x`
  --> examples/demo.jai:7:5
   |
 5 | let x = 1
   |     - `x` declared immutable here
 ...
 7 |     x = 2
   |     ^^^^^ assignment to immutable binding
   |
help: change the declaration to `var x = 1`
```

These are what the codes mean:

| Code | Area |
|---|---|
| `E00xx` | lexical |
| `E01xx` | syntax |
| `E02xx` | names |
| `E03xx` | bindings |
| `E04xx` | types |
| `E05xx` | match |
| `E06xx` | functions |
| `E07xx` | classes |
| `E08xx` | modules |

---

## Architecture

```
source --> lexer --> parser --> resolver --> type checker --> codegen --> VM
            |         │           │              │               │         │
          tokens     AST      symbols +      types +          bytecode   values
                               slots          casts           + caches   + GC
```

## Standard library

| Area | Modules |
|---|---|
| Core | `std.core` `std.math` `std.num` `std.str` `std.fmt` |
| Data | `std.list` `std.dict` `std.set` `std.tuple` `std.iter` `std.func` |
| Structures | `std.collections` `std.ds` (trees, graphs, matrices, tries) |
| Algorithms | `std.algo.sort` `.search` `.dp` `.numtheory` `.geometry` `.stat` `.strings` |
| Text | `std.re` (Thompson NFA, linear time) `std.json` |
| System | `std.io` `std.os` `std.time` `std.random` `std.thread` |
| Compute | `std.gpu` (Metal, with CPU fallback) `std.gui` |
| Testing | `std.test` (assertions, suites, benchmarks) |

The toolchain is written in Jaithon too, but it is not standard library: it sits
beside `std` in its own package.

| Area | Modules |
|---|---|
| Toolchain | `jaithon.ast` `jaithon.compile` `jaithon.tool` |

```jai
from std.collections import LRUCache, Counter
from std.algo.search import binary_search
import std.json

let cache = LRUCache[str, int](capacity: 128)
let counts = Counter(words)
print(counts.most_common(5))
print(json.stringify({"ok": true}, indent: 2))
```

## Repository layout

```
jaithon/
├── spec/               LANGUAGE.md, BYTECODE.md, STYLE.md — normative
├── src/
│   ├── common/         allocator, arena, buffers, UTF-8, diagnostics
│   ├── lang/           tokens, lexer, AST, parser
│   ├── sema/           types, resolver, checker, constant folding
│   ├── codegen/        bytecode emission, optimiser, verifier
│   ├── vm/             values, objects, table, chunk, GC, interpreter, .jaic
│   ├── runtime/        primitives, exceptions, methods, module loader
│   ├── native/         threads, process, Metal GUI and GPU
│   └── cli/            argument parsing, REPL
├── lib/
│   ├── std/            the standard library, in Jaithon
│   └── jaithon/        the toolchain, in Jaithon: ast, compile/, tool/
├── tests/
│   ├── golden/         program + expected stdout
│   ├── lang/ stdlib/   unit tests via std.test
│   ├── errors/         each file asserts one diagnostic code
│   └── bench/          Jaithon + Python benchmark pairs
├── examples/           runnable programs
├── editors/vscode/     syntax, snippets, diagnostics, formatter integration
└── scripts/            test, bench, install
```

---

## Contributing

```bash
make debug            # -O0 -g, assertions on
make check            # type-check the whole tree
make test             # full suite
make bootstrap        # differential front-end verification
jaithon fmt --check . # formatting gate
```

The formatter is canonical and has no options, so formatting is never a
discussion. New language behaviour needs a `spec/LANGUAGE.md` change, a golden
test, and — if it can fail — a `tests/errors/` case.

---

## License

MIT. See [LICENSE](LICENSE).

Created by [Abhirama Sonny](https://abhiramasonny.com/).
