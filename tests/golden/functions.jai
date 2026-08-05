#: Functions, defaults, variadics, closures, and higher-order use (spec §6).
fn add(a: int, b: int) -> int {
    return a + b
}
print(add(2, 3))

# Named arguments at the call site.
fn greet(name: str, greeting: str = "Hello") -> str {
    return f"{greeting}, {name}!"
}
print(greet("Ada"))
print(greet("Alan", "Hi"))
print(greet(name: "Grace", greeting: "Hey"))

# Defaults are evaluated per call, so a mutable default is not shared.
fn collect(item: int, into: list[int] = []) -> list[int] {
    into.push(item)
    return into
}
print(collect(1))
print(collect(2))

# Variadic.
fn total(...nums: int) -> int {
    var sum = 0
    for n in nums { sum += n }
    return sum
}
print(total())
print(total(1, 2, 3))
print(total(10, 20, 30, 40))

# Recursion.
fn factorial(n: int) -> int {
    if n < 2 { return 1 }
    return n * factorial(n - 1)
}
print(factorial(10))

# Mutual recursion — declarations are hoisted, so order does not matter.
fn is_even(n: int) -> bool {
    if n == 0 { return true }
    return is_odd(n - 1)
}

fn is_odd(n: int) -> bool {
    if n == 0 { return false }
    return is_even(n - 1)
}
print(is_even(10))
print(is_odd(10))

# Closures capture by reference for `var`, by value for `let`.
fn make_counter() -> fn() -> int {
    var count = 0
    return fn() -> int {
        count += 1
        return count
    }
}
let next = make_counter()
print(next())
print(next())
print(next())

let other = make_counter()
print(other())

# Lambdas.
let square = |x| x * x
print(square(6))

let compose = |f, g| |x| f(g(x))
let add_one = |x| x + 1
print(compose(square, add_one)(3))

# Functions are values.
fn apply_twice(f: fn(int) -> int, x: int) -> int {
    return f(f(x))
}
print(apply_twice(add_one, 5))

# Higher-order builtins.
let nums = [1, 2, 3, 4, 5, 6]
print(nums.map(|x| x * 10))
print(nums.filter(|x| x % 2 == 0))
print(nums.reduce(0, |acc, x| acc + x))
print(nums.sorted(reverse: true))

# void functions.
fn log_it(message: str) -> void {
    print(f"[log] {message}")
}
log_it("done")

# Early return and guard style.
fn safe_div(a: float, b: float) -> float? {
    if b == 0.0 { return null }
    return a / b
}
print(safe_div(1.0, 2.0))
print(safe_div(1.0, 0.0) ?? -1.0)
