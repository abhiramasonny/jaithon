# JAITHON Language Reference

Complete documentation for the JAITHON programming language.

---

## Table of Contents

1. [Basic Syntax](#basic-syntax)
2. [Variables](#variables)
3. [Data Types](#data-types)
4. [Operators](#operators)
5. [Control Flow](#control-flow)
6. [Functions](#functions)
7. [Classes and Objects](#classes-and-objects)
8. [Namespaces](#namespaces)
9. [Arrays](#arrays)
10. [Standard Library](#standard-library)
11. [Built-in Functions](#built-in-functions)
12. [Input/Output](#inputoutput)
13. [Modules](#modules)

---

## Basic Syntax

### Comments

Lines starting with `#` are comments.

```
# This is a comment
var x = 5  # Inline comment
```

### Statements

Each statement is on its own line. No semicolons required.

```
var x = 5
print x
```

---

## Variables

### Declaration

**Syntax:** `var name = value`

```
var x = 42
var pi = 3.14159
var name = "Jaithon"
var flag = true
```

### Assignment

Variables can be reassigned:

```
var x = 5
x = x + 1
print x    # Outputs: 6
```

### Java-style declarations (Java + Python hybrid)

You can also declare variables and functions using a Java-like signature with explicit return/field types. Use `var` as the type when you want Python-style dynamic typing, and optionally place the declaration inside a namespace with `in <namespace>` (it will be created if missing).

```
int counter = 0
long total = 0
float ratio
double piApprox = 3.14
string name = "Abhi"
char initial = "A"
bool isReady = true
byte small = 1
short mid = 1000
var cache in services = loadCache()

public static void main in core()
    print "hello"
end

private var add in math(int a, int b)
    return a + b
end

# Larger types can be layered in modules:
# BigInt can be bootstrapped from these primitives.
```

`public`, `private`, `protected`, and `static` are accepted for readability; they are not enforced at runtime yet but keep your code close to Java signatures.

---

## Data Types

| Type | Description | Example |
|------|-------------|---------|
| Number | Integers and floats (supports scientific notation) | `42`, `3.14`, `-7`, `1e6`, `2.2e-16` |
| String | Text in quotes | `"hello"` |
| Boolean | `true` or `false` (printed as `1`/`0`) | `true`, `false` |
| Null | No value | `null` |
| Function | User-defined function | `func add(a,b) ... end` |
| Array | Dynamic list of values | `[1, 2, 3]`, `[]` |
| Object | Instance of a `class` | `new Dog("Rex", "Lab")` |

---

## Operators

### Arithmetic

| Operator | Description | Example | Result |
|----------|-------------|---------|--------|
| `+` | Addition | `2 + 3` | `5` |
| `-` | Subtraction | `5 - 2` | `3` |
| `*` | Multiplication | `4 * 3` | `12` |
| `/` | Division | `10 / 2` | `5` |
| `%` | Modulo | `7 % 3` | `1` |
| `^` | Exponent | `2 ^ 3` | `8` |
| `!` | Factorial | `5!` | `120` |

### Comparison

| Operator | Description | Example | Result |
|----------|-------------|---------|--------|
| `>` | Greater than | `5 > 3` | `true` |
| `<` | Less than | `2 < 4` | `true` |
| `>=` | Greater or equal | `5 >= 5` | `true` |
| `<=` | Less or equal | `3 <= 4` | `true` |
| `eq` | Equal to | `5 eq 5` | `true` |

### Logical

| Operator | Description | Example | Result |
|----------|-------------|---------|--------|
| `and` | Logical AND | `true and false` | `false` |
| `or` | Logical OR | `true or false` | `true` |
| `not` | Logical NOT | `not true` | `false` |

### Operator Precedence (highest to lowest)

1. `.` (member access), `!` (factorial), `^` (exponent)
2. `*`, `/`, `%`
3. `+`, `-`
4. `>`, `<`, `>=`, `<=`
5. `eq`
6. `and`
7. `or`

All binary operators are left-associative with the above precedence (exponentiation is also left-associative in Jaithon).

---

## Control Flow

### If Statement

**Syntax:**
```
if condition
    statements
end
```

**With else:**
```
if condition
    statements
else
    other_statements
end
```

**Example:**
```
var x = 10
if x > 5
    print "x is big"
else
    print "x is small"
end
```

### While Loop

**Syntax:**
```
while condition
    statements
end
```

**Example:**
```
var i = 0
while i < 5
    print i
    i = i + 1
end
```

### Break

Exit a loop early:

```
var i = 0
while i < 100
    if i > 10
        break
    end
    print i
    i = i + 1
end
```

---

## Functions

### Definition

**Syntax:**
```
func name(param1, param2, ...)
    statements
    return value
end
```

**Example:**
```
func add(a, b)
    return a + b
end

func greet(name)
    print "Hello, " + name
end
```

### Calling Functions

```
var result = add(3, 4)
print result    # Outputs: 7

greet("World")  # Outputs: Hello, World
```

### Return Values

Use `return` to return a value. Functions without return give `null`.

```
func square(x)
    return x * x
end

func sayHi()
    print "Hi!"
    # Returns null implicitly
end
```

---

## Standard Library

The standard library (`lib/std.jai`) is written in Jaithon and loaded automatically. It pulls in modular files from `lib/modules/` (constants, core, math, list, array, string, random, stack, queue, vector, hashmap, etc.). You can import a specific module directly with its short name (e.g. `import array`, `import core/core`, `import gui/window`); the loader resolves names into `lib/modules/**.jai` or `**/index.jai` recursively.

### Data Structures

| Structure | Functions | Notes |
|-----------|-----------|-------|
| Cons list | `cons`, `head`, `tail`, `listLen`, `append`, `map`, `filter`, `foldl` | Built on `_cell`/`_car`/`_cdr` primitives. |
| Arrays    | `arrayNew`, `arrayLen`, `arrayGet`, `arraySet`, `arrayPush`, `arrayPop`, `arrayRange`, `arrayCopy`, `arrayMap`, `arrayFilter`, `arrayReduce` | Dynamic arrays backed by the native array runtime. |
| Stack     | `stack()` constructor with methods `push`, `pop`, `peek`, `isEmpty`, `size` | LIFO stack implemented on top of arrays. |
| Queue     | `queue()` constructor with methods `enqueue`, `dequeue`, `peek`, `isEmpty`, `size` | FIFO queue implemented on top of arrays. |
| Vector    | `vector()` constructor with methods `len`, `get`, `set`, `push`, `pop`, `clear`, `toArray` | Dynamic sequence convenience wrapper over arrays. |

### String Helpers (stdlib)

| Function | Description | Example |
|----------|-------------|---------|
| `repeat(s, n)` | Repeat string | `repeat("ha", 3)` → `"hahaha"` |
| `startsWith(s, prefix)` | Prefix check | `startsWith("hello", "he")` → `true` |
| `endsWith(s, suffix)` | Suffix check | `endsWith("hello", "lo")` → `true` |
| `contains_str(s, sub)` | Substring search | `contains_str("abracadabra", "cada")` → `true` |
| `indexOf(s, sub)` | First index (or -1) | `indexOf("hello", "l")` → `2` |
| `replace(s, old, repl)` | Replace all occurrences | `replace("foo bar foo", "foo", "baz")` → `"baz bar baz"` |
| `reverse_str(s)` | Reverse string | `reverse_str("abc")` → `"cba"` |
| `trim(s)` | Trim whitespace | `trim("  hi  ")` → `"hi"` |
| `upper(s)` / `lower(s)` | Case conversion (ASCII) | `upper("hi")` → `"HI"` |

### GUI (Metal on macOS)

The GUI module (`import gui/window`) exposes a `Window` class and helpers backed by native Metal code (macOS/Apple Silicon only). Drawing primitives are implemented in Jai atop a native pixel buffer; only the bare hardware hooks are native.

| Item | Description |
|------|-------------|
| `rgb(r, g, b)` | Compose 24-bit color |
| `Window` fields | `width`, `height`, `targetFPS` |
| `Window.init(w, h, title)` / `initWithFPS(...)` | Create a window |
| `putPixel(x, y, color)` | Write a pixel into the backbuffer |
| `clear(color)` | Clear backbuffer |
| `fillRect`, `drawRect`, `drawLine`, `drawCircle`, `fillCircle` | Drawing primitives |
| `display()` / `update()` | Request + present backbuffer (vsync) |
| `present()` | Convenience: `display` + `update` |
| `getDeltaTime()` / `getFPS()` / `getTime()` | Timing utilities |
| `poll()` | Pump events |
| `mousePos()` | `[x, y]` array of mouse position |
| `mouseDown(button)` | `true/false` for button 0/1/2 |
| `keyDown(keycode)` | `true/false` for a key code |

Keyboard helpers live in `lib/modules/gui/keyboard.jai` with macOS keycode constants plus `keyboardTick(win)` which returns an array of keys newly pressed since the last tick.

### Math Utilities

| Function | Description | Example |
|----------|-------------|---------|
| `max(a, b)` | Maximum | `max(5, 10)` → `10` |
| `min(a, b)` | Minimum | `min(5, 10)` → `5` |
| `clamp(v, lo, hi)` | Clamp to range | `clamp(15, 0, 10)` → `10` |
| `sign(x)` | Sign (-1, 0, 1) | `sign(-5)` → `-1` |
| `pow2(x)` | x² | `pow2(4)` → `16` |
| `pow3(x)` | x³ | `pow3(3)` → `27` |
| `fact(n)` | Factorial | `fact(5)` → `120` |
| `gcd(a, b)` | GCD | `gcd(48, 18)` → `6` |
| `lcm(a, b)` | LCM | `lcm(4, 6)` → `12` |
| `sumTo(n)` | Sum 1 to n | `sumTo(10)` → `55` |
| `avg(a, b)` | Average | `avg(4, 6)` → `5` |

### Boolean Utilities

| Function | Description | Example |
|----------|-------------|---------|
| `isEven(n)` | Is even? | `isEven(6)` → `true` |
| `isOdd(n)` | Is odd? | `isOdd(7)` → `true` |
| `between(v, lo, hi)` | v in (lo, hi)? | `between(5, 0, 10)` → `true` |
| `inRange(v, lo, hi)` | v in [lo, hi]? | `inRange(5, 0, 10)` → `true` |

### File I/O (io module)

The `io` module provides functions for reading and writing files.

| Function | Description | Example |
|----------|-------------|---------|
| `io.readFile(path)` | Read entire file content | `var content = io.readFile("data.txt")` |
| `io.writeFile(path, content)` | Write content to file | `io.writeFile("log.txt", "Error")` |
| `io.open(path, mode)` | Open file ("r", "w", "a") | `var f = io.open("test.txt", "r")` |
| `io.close(file)` | Close file handle | `io.close(f)` |
| `io.read(file)` | Read from handle | `var s = io.read(f)` |
| `io.write(file, content)` | Write to handle | `io.write(f, "hello")` |

---

## Built-in Functions

These are implemented in C for performance:

### Math

| Function | Description |
|----------|-------------|
| `sin(x)` | Sine (radians) |
| `cos(x)` | Cosine (radians) |
| `tan(x)` | Tangent (radians) |
| `sqrt(x)` | Square root |
| `abs(x)` | Absolute value |
| `floor(x)` | Round down |
| `ceil(x)` | Round up |
| `round(x)` | Round to nearest |

### Utility

| Function | Description |
|----------|-------------|
| `time()` | Current time (seconds since epoch) |
| `rand()` | Random number 0.0 to 1.0 |
| `len(x)` | Length (arrays or strings) |
| `str(x)` | Convert to string |
| `num(s)` | Convert to number |
| `type(x)` | String name of the value's type |
| `_charAt(s, i)` | Character at index |
| `_substr(s, start, count)` | Substring |
| `_concat(a, b)` | Concatenate strings |

### Cons Cells (Data Structures)

Jaithon provides cons cells as the primitive for building data structures like linked lists:

| Function | Description |
|----------|-------------|
| `_cell(car, cdr)` | Create a cons cell with car and cdr values |
| `_car(cell)` | Get the car (first) value of a cell |
| `_cdr(cell)` | Get the cdr (second) value of a cell |
| `_setcar(cell, val)` | Set the car value (mutates in place) |
| `_setcdr(cell, val)` | Set the cdr value (mutates in place) |

Example:
```
var pair = _cell(1, 2)
print _car(pair)      # Outputs: 1
print _cdr(pair)      # Outputs: 2

# Build a linked list: (1 -> 2 -> 3 -> null)
var list = _cell(1, _cell(2, _cell(3, null)))
```

The standard library (`std.jai`) provides higher-level list functions built on cons cells:
- `cons(x, list)` - prepend element
- `head(list)` / `tail(list)` - first element / rest
- `listLen(list)` - length
- `listGet(list, i)` / `listSet(list, i, v)` - indexed access
- `range(a, b)` - create list of numbers from a to b-1
- `map(f, list)` / `filter(f, list)` / `foldl(f, acc, list)` - functional operations

### Constants

| Constant | Value |
|----------|-------|
| `PI` | 3.14159265358979 |
| `E` | 2.71828182845905 |

---

## Input/Output

### Print

```
print 42
print "Hello"
print x + y
```

### Input

```
input name
print "Hello, " + name
```

---

## Modules

### Import

Import another `.jai` file. Use `/` to navigate subdirectories (relative to the current working directory); `.jai` is added automatically.

```
import calculator           # loads calculator.jai
import math     # loads math module from lib/modules recursively
```
All imports are resolved from the process working directory; they are not relative to the importing file.

The standard library aggregates modules under `lib/modules/` via `lib/std.jai`. Current modules include:

- `constants` (PI, E, etc.)
- `core` (wrappers for native helpers: `len`, `str`, `num`, `type`, math delegates)
- `math`, `list`, `array`, `string`, `random`
- `ds/array`, `ds/stack`, `ds/queue`, `ds/vector`, `ds/linkedlist`, `ds/btree`, `ds/hashmap`
- `gui/window`, `gui/keyboard` (Metal-only GUI + input)
- `io`

Imports are parsed in an isolated module namespace; exported values/functions are then injected into the caller unless a name already exists (caller definitions win). Errors report the module path and line number, and a call stack is printed for easier debugging.

### System Commands

Execute shell commands:

```
system "ls -la"
```

---

## Grammar (EBNF)

```ebnf
program     = { statement } ;

statement   = varDecl
            | printStmt
            | ifStmt
            | whileStmt
            | funcDecl
            | classDecl
            | returnStmt
            | importStmt
            | inputStmt
            | breakStmt
            | delStmt
            | systemStmt
            | assignment
            | funcCall ;

varDecl     = "var" IDENTIFIER "=" expression ;
printStmt   = "print" expression ;
ifStmt      = "if" expression [ "then" ] [ "do" ] { statement } [ "else" { statement } ] "end" ;
whileStmt   = "while" expression [ "do" ] { statement } "end" ;
funcDecl    = "func" IDENTIFIER "(" [ params ] ")" { statement } "end" ;
classDecl   = "class" IDENTIFIER [ "extends" IDENTIFIER ] { classMember } "end" ;
classMember = varDecl | methodDecl ;
methodDecl  = "func" IDENTIFIER "(" [ params ] ")" { statement } "end" ;
returnStmt  = "return" [ expression ] ;
importStmt  = "import" modulePath ;
modulePath  = IDENTIFIER { "/" IDENTIFIER } ;
inputStmt   = "input" IDENTIFIER ;
breakStmt   = "break" ;
delStmt     = "del" IDENTIFIER [ "[" expression "]" ] ;
systemStmt  = "system" STRING ;
assignment  = IDENTIFIER "=" expression ;
funcCall    = IDENTIFIER "(" [ args ] ")" ;

params      = param { "," param } ;
param       = ["*"] IDENTIFIER ;
args        = expression { "," expression } ;

expression  = orExpr ;
orExpr      = andExpr { "or" andExpr } ;
andExpr     = eqExpr { "and" eqExpr } ;
eqExpr      = compExpr { "eq" compExpr } ;
compExpr    = addExpr { ( ">" | "<" | ">=" | "<=" ) addExpr } ;
addExpr     = mulExpr { ( "+" | "-" ) mulExpr } ;
mulExpr     = powExpr { ( "*" | "/" | "%" ) powExpr } ;
powExpr     = unaryExpr { "^" unaryExpr } ;
unaryExpr   = [ "-" | "not" ] factExpr ;
factExpr    = primary [ "!" ] ;
primary     = NUMBER | STRING | IDENTIFIER | "self" | "true" | "false" | "null"
            | "(" expression ")" | funcCall | arrayLit | newExpr ;

arrayLit    = "[" [ expression { "," expression } ] "]" ;
newExpr     = "new" IDENTIFIER "(" [ args ] ")" ;

IDENTIFIER  = letter { letter | digit | "_" } ;
NUMBER      = digit { digit } [ "." digit { digit } ] [ ( "e" | "E" ) [ "+" | "-" ] digit { digit } ] ;
STRING      = '"' { character } '"' ;
```

---

## Arrays

### Array Literals

Create arrays with square bracket syntax:

```
var arr = [1, 2, 3, 4, 5]
var names = ["Alice", "Bob", "Charlie"]
var empty = []
```

### Indexing

Access elements by index (0-based):

```
var arr = [10, 20, 30]
print arr[0]    # 10
print arr[2]    # 30
arr[1] = 99     # Modify element
```

### Nested Arrays

```
var matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print matrix[1][1]    # 5
```

### Array Functions

| Function | Description | Example |
|----------|-------------|---------|
| `_alen(arr)` | Get array length | `_alen([1,2,3])` → `3` |
| `_push(arr, val)` | Add to end | `_push(arr, 4)` |
| `_pop(arr)` | Remove from end | `_pop(arr)` |
| `_get(arr, i)` | Get element | `_get(arr, 0)` |
| `_set(arr, i, val)` | Set element | `_set(arr, 0, 99)` |

Idiomatic helpers live in `lib/modules/ds/array.jai` and mirror the above with friendlier names (`arrayLen`, `arrayPush`, `arrayPop`, `arrayGet`, `arraySet`, etc.).

---

## Classes and Objects

### Class Definition

```
class ClassName
    var field1
    var field2
    
    func init(self, param1, param2)
        self.field1 = param1
        self.field2 = param2
    end
    
    func method(self)
        return self.field1 + self.field2
    end
end
```

### Creating Objects

Use `new` to instantiate:

```
var obj = new ClassName(arg1, arg2)
```

### Field Access

```
print obj.field1
obj.field1 = newValue
```

### Method Calls

```
obj.method()
var result = obj.method()
```

When a method signature's first parameter is named `self`, calling via `obj.method()` automatically supplies `self` for you.

### Inheritance

Use `extends` to inherit from a parent class:

```
class Animal
    var name
    func init(self, name)
        self.name = name
    end
    func speak(self)
        print self.name + " makes a sound"
    end
end

class Dog extends Animal
    var breed
    func init(self, name, breed)
        self.name = name
        self.breed = breed
    end
    func speak(self)
        print self.name + " barks!"
    end
end

var d = new Dog("Buddy", "Golden Retriever")
d.speak()    # Buddy barks!
```

---

## Namespaces

Namespaces allow you to group variables and functions under a named scope. This helps organize code and avoid naming conflicts.

### Definition

Use the `namespace` keyword:

```
namespace Math
    var PI = 3.14159
    
    func area(r)
        return PI * r * r
    end
end
```

### Accessing Members

Use dot notation to access variables and functions inside a namespace:

```
print Math.PI
print Math.area(10)
```

### Modifying Variables

You can modify variables inside a namespace:

```
Math.PI = 3.14
```

Functions inside the namespace can also modify namespace variables, and the changes persist.

---

## Variadic Functions

Functions that accept any number of arguments:

```
func sum(*args)
    var total = 0
    var i = 0
    while i < _alen(args)
        total = total + args[i]
        i = i + 1
    end
    return total
end

print sum(1, 2, 3)           # 6
print sum(10, 20, 30, 40)    # 100
```

---

## Example Programs

### Hello World

```
print "Hello, World!"
```

### Fibonacci

```
input n
var a = 0
var b = 1
var i = 0
while i < n
    print a
    var temp = a + b
    a = b
    b = temp
    i = i + 1
end
```

### Factorial Function

```
func factorial(n)
    if n < 2
        return 1
    end
    return n * factorial(n - 1)
end

print factorial(5)    # Outputs: 120
```

### Stack Implementation

```
class Stack
    var items
    var size
    
    func init(self)
        self.items = []
        self.size = 0
    end
    
    func push(self, val)
        _push(self.items, val)
        self.size = self.size + 1
    end
    
    func pop(self)
        self.size = self.size - 1
        return _pop(self.items)
    end
end

var s = new Stack()
s.push(10)
s.push(20)
print s.pop()    # 20
```

### Using Standard Library

```
print max(10, 20)           # 20
print min(10, 20)           # 10
print fact(6)               # 720
print gcd(48, 18)           # 6
print clamp(25, 0, 10)      # 10
print isEven(42)            # true

---

## Quick Standard Library Guide

### Math & Number Utilities
- `max/min`, `max3/min3`, `clamp`, `sign`
- Powers and roots: `pow`, `pow2`, `pow3`, `sqrt`, `cbrt`, `hypot`
- Number theory: `gcd`, `lcm`, `isPrime`, `nextPrime`, `prevPrime`
- Combinatorics: `fact`, `nPr`, `nCr`
- Randomness: `rand`, `randInt(lo, hi)`, `randFloat(lo, hi)`, `randBool()`, `randChoice(arr)`, `shuffle(arr)`
- Angles: `degToRad`, `radToDeg`

### Strings
- Basics: `len`, `charAt`, `substr`, `concat`, `repeat`
- Queries: `startsWith`, `endsWith`, `contains_str`, `indexOf`
- Case helpers: `upper`, `lower`
- Replace/trim: `replace(hay, needle, repl)`, `trim`

### Arrays (bootstrapped in `lib/modules/ds/array`)
- Creation: `arrayFill(n, v)`, `arrayRange(start, stop)`
- Accessors: `arrayLen`, `arrayGet`, `arraySet`
- Bulk ops: `arrayConcat`, `arrayCopy`, `arraySlice`, `arrayReverse`
- Higher order: `arrayMap(arr, f)`, `arrayFilter(arr, pred)`, `arrayReduce(arr, init, f)`
- Queries: `arrayContains`, `arrayIndexOf`, `arrayMax`, `arrayMin`, `arraySum`

### Data Structures
- `Stack`, `Queue`, `Vector`, `LinkedList`, `BST`, `HashMap` (all defined in Jai; inspect `lib/modules/ds/` for methods).

### Files & IO
- `open(path, mode)`, `read(handle)`, `write(handle, data)`, `close(handle)`
- Helpers: `readFile(path)`, `writeFile(path, data)`

---

## Namespaces and Modules Patterns

- **Java-style declarations**:  
  `public double area in Circle(double r)` declares a function inside namespace `Circle`. The namespace is created on demand.
- **Block namespaces**:  
  ```
  namespace Metrics
      var scale = 2
      func mult(x) return x * scale end
  end
  ```
- **Imports**: `import ds/array` brings helpers into the current module. Caller-defined symbols win if names collide.
- **Stateful namespaces**: Keep namespace state in module-level variables and expose setters/getters to avoid accidental shadowing.

---

## Classes & Objects: Tips

- Constructors are plain methods named `init(self, ...)` and are called automatically by `new`.
- Methods always receive `self` when invoked with dot notation: `obj.method(1, 2)`.
- Inheritance: `class Dog extends Animal` copies fields/methods; override by re-declaring the method.
- Namespaced classes: you can declare `class Foo in Bar` using Java-style syntax if you want class names under a namespace.

Example with composition and override:
```
class Animal
    var name
    func init(self, n) self.name = n end
    func sound(self) return "..." end
end

class Dog extends Animal
    func sound(self) return self.name + " barks" end
end

var d = new Dog("Rex")
print d.sound()
```

---

## Control Flow Edge Cases

- `and` / `or` are short-circuiting. Only evaluate the right-hand side if needed.
- `break` exits the nearest loop; nested loops require an outer condition to stop both.
- `del x` removes a binding like Python; `del arr[i]` deletes the element and shifts the list left, erroring on out-of-bounds.
- Factorial `!` binds tightly: `2 * 3!` is `2 * (3!)`.

---

## Parallel & GPU (overview)

- CLI flag `--threads=N` caps worker threads (auto-detect by default).
- Parallel library entrypoints live in `lib/modules/parallel/*` (vectorized loops, reductions). When enabled, bytecode is analyzed and parallelized where safe; debug output reports how many loops were parallelized.
- GPU path (Metal on macOS) is enabled by default for GUI and some parallel ops; disable with `--no-gpu`.

---

## GUI Quickstart (Metal)

Minimal window:
```
import gui/window

func main()
    var win = new Window()
    win.init(400, 300, "Hello")
    while true
        win.poll()
        win.clear(rgb(30, 30, 40))
        win.fillRect(180, 130, 40, 40, rgb(80, 200, 255))
        win.display()
        if not win.update() break end
    end
end

main()
```
Keyboard helpers live in `lib/modules/gui/keyboard`; call `keyboardTick(win)` each frame to get newly-pressed keys.

---

## Testing & CLI Workflow

- Run a file: `./jaithon path/to/file.jai`
- Debug mode: `./jaithon -d file.jai` (prints symbol binds and timing).
- REPL: `./jaithon`
- Headless stdlib only: `./jaithon --no-stdlib` if you want a minimal environment.
- Test suite samples live in `test/checks/`; add a file there and run it directly. Expected outputs are documented in comments at the top of each test.

---

## Troubleshooting

- `VM Error: Cannot index non-array` usually means you indexed a non-array value; check you didn't shadow your array variable with a number.
- Namespaces: prefer the Java-style `in Namespace` declarations if you need to call functions as `Namespace.fn()` without warnings.
- GUI: if nothing draws, ensure `win.display()` is called before `win.update()` and that the loop calls `win.poll()` each frame.
```
