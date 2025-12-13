# JAITHON

**A beginner-friendly, mostly-bootstrapped programming language**

[Documentation](docs/grammar.md) • [Author](https://abhiramasonny.com/)

---

## About

Jaithon combines Java-like classes with Python-like syntax, while keeping most of the standard library and many features bootstrapped in Jai itself.

---

## Quick Start

### Prerequisites

- GCC compiler
- Make build tool
- readline library

### Installation

```bash
git clone https://github.com/abhiramasonny/jaithon
cd jaithon
make
```

### macOS install to system paths

```bash
./scripts/install_macos.sh   # copies jaithon to /usr/local/bin and libs to /usr/local/share/jaithon
```

### Running Programs

```bash
# Run a file
./jaithon test/checks/variables.jai

# Enter interactive shell
./jaithon

# View options
./jaithon -h
```

### Module search paths

Imports are resolved from the current directory, `$JAITHON_LIB`, the Jaithon install dir, and common macOS prefixes (`/usr/local/share/jaithon`, `/usr/local/lib/jaithon`, `/Library/Jaithon`, `/opt/homebrew/share/jaithon`).

### Command Line Options

| Option | Description |
|--------|-------------|
| `-d, --debug` | Enable debug mode |
| `-s, --shell` | Force shell mode |
| `-v, --version` | Display version |
| `-h, --help` | Show help |
| `-N, --no-stdlib` | Don't load standard library |
| `--no-extension` | Don't auto-append .jai extension |

---

## Language Features (at a glance)

- Clear syntax: variables, math, logic, if/else, while, functions (including variadic), namespaces.
- Arrays with literals, indexing, and helpers (`_alen`, `_push`, `_pop`) plus higher-level `lib/modules/ds/array`.
- Classes/objects with inheritance; methods auto-receive `self` when called via dot.
- Modules/imports; system command execution via `system "cmd"`.
- Bootstrapped stdlib: math/string helpers, random, data structures (stack/queue/vector/linkedlist/hashmap/btree), file I/O.
- GUI module (Metal on macOS): windows, drawing primitives, timing, mouse/keyboard input, and keycode helpers.

See `docs/grammar.md` for the full language reference and API details.

---

## Standard Library

Jaithon includes a **bootstrapped standard library** written in Jaithon itself (`lib/std.jai`). These functions are available automatically:

### Math Functions

| Function | Description | Example |
|----------|-------------|---------|
| `max(a, b)` | Maximum of two values | `max(5, 10)` → 10 |
| `min(a, b)` | Minimum of two values | `min(5, 10)` → 5 |
| `max3(a, b, c)` | Maximum of three | `max3(1, 5, 3)` → 5 |
| `min3(a, b, c)` | Minimum of three | `min3(1, 5, 3)` → 1 |
| `clamp(v, lo, hi)` | Clamp value to range | `clamp(15, 0, 10)` → 10 |
| `sign(x)` | Sign of number (-1, 0, 1) | `sign(-5)` → -1 |
| `pow(b, e)` | Power function | `pow(2, 8)` → 256 |
| `pow2(x)` / `pow3(x)` / `pow4(x)` | Square/Cube/Fourth power | `pow2(4)` → 16 |

### Number Properties

| Function | Description | Example |
|----------|-------------|---------|
| `isEven(n)` / `isOdd(n)` | Parity check | `isEven(6)` → true |
| `isPositive(n)` / `isNegative(n)` / `isZero(n)` | Sign check | `isPositive(5)` → true |
| `isDivisible(n, d)` | Divisibility check | `isDivisible(10, 5)` → true |

### Combinatorics

| Function | Description | Example |
|----------|-------------|---------|
| `fact(n)` | Factorial | `fact(5)` → 120 |
| `permutations(n, r)` | nPr | `permutations(5, 2)` → 20 |
| `combinations(n, r)` | nCr | `combinations(10, 3)` → 120 |

### Number Theory

| Function | Description | Example |
|----------|-------------|---------|
| `gcd(a, b)` | Greatest common divisor | `gcd(48, 18)` → 6 |
| `lcm(a, b)` | Least common multiple | `lcm(4, 6)` → 12 |
| `fib(n)` | Fibonacci number | `fib(10)` → 55 |
| `isPrime(n)` | Primality test | `isPrime(17)` → true |

### Series & Sums

| Function | Description | Example |
|----------|-------------|---------|
| `sumTo(n)` | Sum 1 to n | `sumTo(10)` → 55 |
| `sumFromTo(a, b)` | Sum a to b | `sumFromTo(5, 10)` → 45 |
| `sumSquares(n)` | Sum of squares 1² to n² | `sumSquares(3)` → 14 |
| `sumCubes(n)` | Sum of cubes 1³ to n³ | `sumCubes(3)` → 36 |

### Geometry

| Function | Description | Example |
|----------|-------------|---------|
| `distance2D(x1, y1, x2, y2)` | Distance between points | `distance2D(0, 0, 3, 4)` → 5 |
| `hypotenuse(a, b)` | Pythagorean hypotenuse | `hypotenuse(3, 4)` → 5 |
| `areaCircle(r)` | Circle area | `areaCircle(5)` → 78.54 |
| `areaRect(w, h)` | Rectangle area | `areaRect(4, 5)` → 20 |
| `areaTriangle(b, h)` | Triangle area | `areaTriangle(6, 4)` → 12 |

### Conversions

| Function | Description | Example |
|----------|-------------|---------|
| `degToRad(d)` / `radToDeg(r)` | Angle conversion | `degToRad(180)` → 3.14 |
| `celsiusToFahr(c)` / `fahrToCelsius(f)` | Temperature | `celsiusToFahr(0)` → 32 |

### Utilities

| Function | Description |
|----------|-------------|
| `avg(a, b)` / `avg3(a, b, c)` | Average |
| `between(v, lo, hi)` | Check if v in (lo, hi) exclusive |
| `inRange(v, lo, hi)` | Check if v in [lo, hi] inclusive |
| `approxEq(a, b, eps)` | Approximate equality |
| `double(x)` / `half(x)` | Double or halve |
| `inc(x)` / `dec(x)` | Increment or decrement |
| `negate(x)` / `reciprocal(x)` | Negate or invert |

### Array Functions (Bootstrapped)

| Function | Description | Example |
|----------|-------------|---------|
| `array(n)` | Create array of size n | `var a = array(5)` |
| `get(arr, i)` | Get element at index | `get(a, 0)` |
| `set(arr, i, v)` | Set element at index | `set(a, 0, 10)` |
| `push(arr, v)` | Append element | `push(a, 42)` |
| `pop(arr)` | Remove and return last | `pop(a)` |
| `arraySum(arr)` | Sum all elements | `arraySum(a)` |
| `arrayMax(arr)` / `arrayMin(arr)` | Find max/min | `arrayMax(a)` |
| `arrayReverse(arr)` | Reverse in place | `arrayReverse(a)` |
| `arrayContains(arr, v)` | Check if contains | `arrayContains(a, 5)` |

### String Functions (Bootstrapped)

| Function | Description | Example |
|----------|-------------|---------|
| `charAt(s, i)` | Get character at index | `charAt("hello", 0)` → "h" |
| `substr(s, i, n)` | Substring | `substr("hello", 0, 3)` → "hel" |
| `concat(a, b)` | Concatenate strings | `concat("hi", "!")` → "hi!" |
| `stringReverse(s)` | Reverse string | `stringReverse("abc")` → "cba" |
| `stringContains(s, sub)` | Check if contains | `stringContains("hello", "ell")` |
| `stringRepeat(s, n)` | Repeat n times | `stringRepeat("ab", 3)` → "ababab" |

### Trig & Advanced Math (Bootstrapped)

| Function | Description |
|----------|-------------|
| `sinh(x)`, `cosh(x)`, `tanh(x)` | Hyperbolic functions |
| `asin(x)`, `acos(x)`, `atan(x)` | Inverse trig |
| `log10(x)`, `log2(x)` | Logarithms |
| `cbrt(x)`, `nthRoot(x, n)` | Cube root, nth root |
| `nextPrime(n)`, `prevPrime(n)` | Prime navigation |
| `catalan(n)`, `lucas(n)` | Special sequences |

### Native Primitives (C Implementation)

These are the only functions implemented in C - everything else is bootstrapped in Jaithon:

| Function | Description |
|----------|-------------|
| `_sin(x)`, `_cos(x)`, `_tan(x)` | Hardware trig |
| `_sqrt(x)`, `_log(x)`, `_exp(x)` | Hardware math |
| `_time()`, `_rand()` | System calls |
| `_len(x)`, `_str(x)`, `_num(x)`, `_type(x)` | Type primitives |
| `_array(n)`, `_get()`, `_set()`, `_push()`, `_pop()` | Array primitives |
| `_charAt()`, `_substr()`, `_concat()` | String primitives |

---

## Project Structure

```
jaithon/
├── src/                    # C source code
│   ├── cli/               # Command-line interface
│   │   └── main.c         # Entry point
│   ├── core/              # Runtime core
│   │   ├── runtime.c      # Values, modules, variables
│   │   └── runtime.h
│   └── lang/              # Language implementation
│       ├── lexer.c/.h     # Tokenizer
│       └── parser.c/.h    # Parser and execution
├── lib/                   # Standard library (bootstrapped)
│   └── std.jai           # Written in Jaithon itself
├── docs/         # Language documentation
├── test/                  # Example programs
│   ├── Jaithon/          # Jaithon test files
│   └── Python/           # Equivalent Python files
├── Extensions/            # VS Code syntax highlighting
└── Makefile
```

---

## VS Code Extension

Install syntax highlighting for `.jai` files:

```bash
cp -r Extensions/jai ~/.vscode/extensions
```

---

## Examples

Great starting points live in `examples/`:

| File | Description |
|------|-------------|
| `graph_paths.jai` | DFS/BFS path tracing |
| `gui_snake.jai` | Simple GUI game built with the Metal module |
| `todo.jai` | Minimal CLI task list showcasing arrays and functions |
| `vector_demo.jai` | Dynamic vector operations |
| `stack_queue_demo.jai` | Stack and queue utilities in one file |

---

## Documentation

Full language reference: [`docs/grammar.md`](docs/grammar.md)

---

## License

See [LICENSE](LICENSE) for details.

---

## Author

Created by [Abhirama Sonny](https://abhiramasonny.com/)
