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

There is one gap, and it is worth knowing exactly where it is. A value routed
through `any` escapes the checker, because with an `any` receiver the checker
does not know the class and so cannot know what a field was declared as. **Field
writes are checked at run time**, so this raises `TypeError`:

```jai
class Point { pub var x: float
              fn init(self, x: float) { self.x = x } }

fn poison(p: any) { p.x = "not a float" }
poison(Point(1.5))          # TypeError: cannot assign str to field 'x' declared float
```

**Container elements are not**, so this is currently accepted:

```jai
let xs: list[int] = []
fn poison2(l: any) { l.push("str") }
poison2(xs)                 # xs is now ["str"]
```

Treat a container's element type as checked at the boundaries the checker can
see, and not through `any`.

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

A type argument can be an integer literal. The checker keeps it as a constant,
which is how a tensor names its shape:

```jai
fn logits(x: Tensor[float, 32, 10]) -> Tensor[float, 32, 10] {
    return x
}
```

`Tensor` is a class (`jaitensor.Tensor` or an import alias), not a builtin.
User generics normally drop their arguments; the checker keeps them when the
name is `Tensor`. Two written shapes have to agree, including the dtype in the
first slot. Bare `Tensor` still assigns both ways, which is the escape hatch
when the rank is not known yet. Storage is float32, so write `float` there.

```jai
let a: Tensor[float, 2, 3] = t
let b: Tensor[float, 2, 4] = a    # type error: dims 3 and 4
```

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

A function can carry `@name` decorators. They are part of the function object,
not comments.

`@trace` is for hot numerical kernels. The JIT compiles that function after 8
calls instead of 64. While it runs, `std.trace.record_op` can record op names
and shapes; a later call with the same sequence of shapes sets
`std.trace.is_replay()`. The outermost `@trace` function owns the graph.
`is_replay()` is a flag for libraries to skip rebuilding a plan; it does not
skip the interpreter by itself. A tail call from a traced function does not
reuse the frame, so the trace session is still closed on return.

```jai
import std.trace as trace

@trace
fn step(x: Tensor, w: Tensor) -> Tensor {
    trace.record_op("matmul", x.shape)
    return matmul(x, w)
}
```

`@gpu_kernel` and `@mps_graph` are the same flag. They mark a function as a
device kernel. `std.gpu.is_gpu_kernel(f)` is how you ask. Compiling Metal still
goes through `gpu.Kernel.compile(source, entry)` with MSL source — the Jaithon
body is not lowered to Metal yet.

```jai
import std.gpu as gpu

const SAXPY = """
#include <metal_stdlib>
using namespace metal;
kernel void saxpy(device float *x [[buffer(0)]],
                  device float *y [[buffer(1)]],
                  constant int &n [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if (i < uint(n)) y[i] = y[i] + x[i];
}
"""

@gpu_kernel
fn saxpy(x: gpu.Buffer, y: gpu.Buffer) -> void {
    let k = gpu.Kernel.compile(SAXPY, "saxpy")
    k.dispatch([x, y], x.count, scalars: [x.count])
}
```

A name that is not `trace`, `gpu_kernel`, or `mps_graph` is stored and
currently ignored.

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

## GPU

`std.gpu` is the language's GPU: device buffers, Metal kernels, and the vector
ops most programs want. Every operation also has a CPU path for when there is
no device.

A `Buffer` holds float32 slots on the device. Device memory is not garbage
collected; call `free` (usually from `defer`). `set_device` has to run before
the first buffer or kernel is created. After that the process keeps that
device.

```jai
import std.gpu as gpu

if gpu.is_available() {
    gpu.set_device(0)
    let buf = gpu.Buffer(1024)
    defer { buf.free() }
    gpu.set_mixed_precision(true)
}
```

`gpu.device_count()` is how many Metal devices the process can see. Mixed
precision runs GEMM and fused MLP graphs in float16 and writes float32.
Storage stays float32.

`Kernel.compile` interns by source and entry name. Freeing an interned kernel
and compiling the same source again recompiles it.

## Machine learning

Training is ordinary Jaithon. `jaitensor` is a workspace package written in
`.jai`: GPU-resident tensors, autograd, Keras-style `Sequential` models, and
the ops a CNN or attention block needs. It queues Metal through `std.gpu` and
marks its hot kernels `@trace`.

```jai
import jaitensor as jt

let data = jt.Dataset.from_rows([[0.0, 1.0], [1.0, 0.0]], [0, 1])
defer { data.free() }

let model = jt.Sequential([
    jt.Dense(128, activation: jt.Activation.Relu),
    jt.Dense(10, activation: jt.Activation.Softmax),
])
defer { model.free() }

model.compile(jt.Adam(), mixed_precision: true)
model.fit(data, epochs: 5, batch_size: 2048, shuffle: true)
```

Images are NHWC. Layers include `Dense`, `Conv2d`, `Flatten`, `BatchNorm2d`,
`GroupNorm2d`, `LayerNorm`, `MultiHeadAttention`, `Dropout`, `Embedding`,
`MaxPool2d`, and `AvgPool2d`. Activations include ReLU, GELU, SiLU, and leaky
ReLU. `DataLoader` can shuffle by gathering rows on the GPU. `fit(..., shuffle:
true)` uses that path. Tensor ops cover the PyTorch-style surface: `abs`,
`sum`/`mean`, `clamp`, `cat`, pooling, `dropout`, `mse`, `eye`, `flip`, nearest
upsample, `tril`, and `binary_cross_entropy_with_logits`. `Embedding` gathers
rows and scatters gradients back into the table. `MultiHeadAttention` with
`heads > 1` uses a tiled flash kernel on long sequences (`head_dim <= 64`)
so the score matrix never hits device memory; other shapes use MMA/MPS GEMM.

A tensor that you allocate yourself needs `free`. Models and datasets free the
tensors they own.

The VM is single-threaded. Closures cannot run on `std.thread` workers, so the
loader overlaps the next gather with GPU work on the same Metal queue rather
than on a host prefetch thread.

`set_device` / `device_count` pick which GPU later allocations use. There is
no built-in all-reduce.

Worked examples: [`mnist_gpu.jai`](examples/mnist_gpu.jai),
[`fashion_mnist.jai`](examples/fashion_mnist.jai),
[`spiral_classifier.jai`](examples/spiral_classifier.jai). The type and
runtime contracts are in [`ml.md`](ml.md).

## Running it

```
jaithon run program.jai      # compile and run
jaithon check program.jai    # type check only
jaithon fmt program.jai      # format in place
jaithon test tests/          # run tests
jaithon repl                 # interactive prompt
jaithon disasm program.jai   # bytecode listing
```
