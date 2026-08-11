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

PS: A majority of the documentation (including documentation within .jai files and .c files via docstrings) are AI generated, and as such might be wrong about certain things. This was done to speed up development, however this is being re-written as I continuiously develop. Currently my focus is on improving jaithon as a language, making it faster etc, so shortcuts like this help me push developments quicker. The readme is completely written by me, same with a majority of the language.md file. I have checked over the language.md file and there are no errors within it.

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

```python
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

more idepth file -> [`LANGUAGE.md`](LANGUAGE.md).

Also you can checkout the examples directory.

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

## Contributing

```bash
make debug            # -O0 -g, assertions on
make check            # type-check the whole tree
make test             # full suite
make bootstrap        # differential front-end verification
jaithon fmt --check . # formatting gate
```

## License

MIT. See [LICENSE](LICENSE).

Created by [Abhirama Sonny](https://abhiramasonny.com/).
