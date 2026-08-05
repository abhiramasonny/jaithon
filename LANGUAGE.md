# The Jaithon language

## Hello

```jai
print("Hello, world!")
```

Save it as `hello.jai` and run `jaithon run hello.jai`.

## Comments

```jai
# A line comment.

#* A block comment,
   which can span lines. *#
```

## Variables

`let` is immutable. `var` is mutable.

```jai
let name = "Ada"
var count = 0
count += 1
```

A type annotation is optional, however when you write one it is enforced by Jaithon.

```jai
let pi: float = 3.14159
var total: int = 0
```

## Types

`int`, `float`, `bool`, `str`, `bytes`, and `null`. Containers are `list`,
`dict`, `set`, `tuple`, and `range`. The type `any` accepts anything and is
checked when it flows into something more specific.

An `int` becomes a `float` where a float is wanted, and the value really is
converted:

```jai
let temperature: float = 20
print(temperature)          # 20.0
```

A float never silently becomes an int, because that would lose the fraction.
Write `int(x)` when you want to truncate.

## Strings

```jai
let greeting = "Hello"
let name = "Ada"
print(f"{greeting}, {name}!")        # Hello, Ada!
print(f"{2 + 2} and {name.upper()}")  # 4 and ADA
```

A string prefixed with `f` interpolates any expression inside braces.

## Collections

```jai
let numbers = [1, 2, 3]
let ages = {"ada": 36, "alan": 41}
let unique = {1, 2, 2, 3}
let pair = (1, "one")

numbers.push(4)
print(numbers.len())        # 4
print(ages["ada"])          # 36
print(numbers[0])           # 1
print(numbers[-1])          # 4
```

Comprehensions build them from a loop:

```jai
print([x * 2 for x in 0..5])            # [0, 2, 4, 6, 8]
print([x for x in 0..10 if x % 3 == 0]) # [0, 3, 6, 9]
print({x: x * x for x in 1..4})         # {1: 1, 2: 4, 3: 9}
```

## Control flow

```jai
let n = 7

if n > 10 {
    print("big")
} elif n > 5 {
    print("medium")
} else {
    print("small")
}
```

`if` is also an expression when both arms produce a value:

```jai
let label = if n % 2 == 0 { "even" } else { "odd" }
```

Loops come in three shapes. `while` tests a condition, `loop` runs until you
`break`, and `for` walks a range or a container.

```jai
var i = 0
while i < 3 { i += 1 }

for k in 0..3 { print(k) }        # 0 1 2
for k in 0..=3 { print(k) }       # 0 1 2 3
for item in ["a", "b"] { print(item) }

for (index, value) in ["x", "y"].enumerate() {
    print(f"{index}:{value}")
}
```

A label lets `break` and `continue` name which loop they mean:

```jai
'outer: for row in 0..5 {
    for col in 0..5 {
        if row * col == 6 { break 'outer }
    }
}
```

`match` compares a value against patterns, in order, and returns the first that
fits:

```jai
fn describe(code: int) -> str {
    return match code {
        200 => "ok",
        301 | 302 => "redirect",
        400..=499 => "client error",
        c if c >= 500 => "server error",
        _ => "unknown",
    }
}
```

## Functions

```jai
fn add(a: int, b: int) -> int {
    return a + b
}
```

Parameters may have defaults, and callers may name them:

```jai
fn greet(name: str, greeting: str = "Hello") -> str {
    return f"{greeting}, {name}!"
}

print(greet("Ada"))
print(greet(name: "Grace", greeting: "Hey"))
```

`...` collects the rest of the arguments:

```jai
fn total(...nums: int) -> int {
    var sum = 0
    for n in nums { sum += n }
    return sum
}
```

A function that returns nothing is written `-> void`. A bare `return` leaves it
early.

Lambdas are written with bars, and functions are values:

```jai
let square = |x| x * x
let nums = [1, 2, 3, 4]
print(nums.map(|x| x * 10))
print(nums.filter(|x| x % 2 == 0))
print(nums.reduce(0, |acc, x| acc + x))
```

A closure captures a `var` by reference and a `let` by value:

```jai
fn make_counter() -> fn() -> int {
    var count = 0
    return fn() -> int {
        count += 1
        return count
    }
}
```

## Classes

Fields are declared. `pub` makes a member visible outside the class; without it
the member is private.

```jai
class Point {
    pub var x: float
    pub var y: float

    fn init(self, x: float, y: float) {
        self.x = x
        self.y = y
    }

    pub fn length(self) -> float {
        return (self.x ** 2 + self.y ** 2) ** 0.5
    }

    pub static fn origin() -> Point {
        return Point(0.0, 0.0)
    }
}

let p = Point(3.0, 4.0)
print(p.length())        # 5.0
print(Point.origin())
```

Methods named `__add__`, `__eq__`, `__str__` and friends give a class its
behaviour under operators and `print`.

A class inherits with `extends`, and the subclass calls `super` to run the
parent's initialiser:

```jai
class Animal {
    pub var name: str

    fn init(self, name: str) { self.name = name }

    pub fn speak(self) -> str { return "..." }
}

class Dog extends Animal {
    pub var breed: str

    fn init(self, name: str, breed: str) {
        super(name)
        self.breed = breed
    }

    pub fn speak(self) -> str { return "Woof" }
}
```

## Traits

A trait states what a type must provide, and may supply defaults.

```jai
trait Shape {
    fn area(self) -> float

    fn name(self) -> str { return "shape" }
}

class Circle: Shape {
    var radius: float

    fn init(self, radius: float) { self.radius = radius }

    fn area(self) -> float { return 3.14159 * self.radius ** 2 }
}
```

## Enums

```jai
enum Color { Red, Green, Blue }

fn hex_of(c: Color) -> str {
    return match c {
        Color.Red => "#ff0000",
        Color.Green => "#00ff00",
        Color.Blue => "#0000ff",
    }
}
```

A variant can carry data, which `match` takes apart:

```jai
enum Expr { Num(value: float), Add(left: Expr, right: Expr) }

fn eval(e: Expr) -> float {
    return match e {
        Expr.Num(v) => v,
        Expr.Add(l, r) => eval(l) + eval(r),
    }
}
```

## Errors

`throw` raises, `try` and `catch` handle. A handler catches subclasses too.

```jai
fn parse_positive(text: str) -> int {
    let parsed = text.to_int()
    if parsed == null { throw ParseError(f"not a number: {text}") }
    let n = parsed ?? 0
    if n <= 0 { throw ValueError(f"expected a positive number, got {n}") }
    return n
}

try {
    print(parse_positive("42"))
    print(parse_positive("-1"))
} catch e: ValueError {
    print(f"ValueError: {e.message}")
}
```

`defer` runs a block when the enclosing function leaves, whether it returns
normally or throws. Deferred blocks run in reverse order.

```jai
fn with_cleanup() {
    defer { print("second") }
    defer { print("first") }
    print("body")
}
```

`??` supplies a value when the left side is null:

```jai
let maybe: int? = null
print(maybe ?? 0)        # 0
```

## Modules

Every file is a module. `pub` decides what leaves it.

```jai
import std.math
from std.math import sqrt
import std.collections as coll
```

A leading dot imports relative to the current file:

```jai
import .helpers
from .helpers import format_row
```

## Running it

```
jaithon run program.jai      # compile and run
jaithon check program.jai    # type check only
jaithon fmt program.jai      # format in place
jaithon test tests/          # run tests
jaithon repl                 # interactive prompt
jaithon disasm program.jai   # bytecode listing
```
