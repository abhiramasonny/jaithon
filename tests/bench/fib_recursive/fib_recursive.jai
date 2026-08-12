#: Deep recursion — measures call-frame setup and teardown.
from std.os import env_or

# Work is exponential in the argument, so the level subtracts from it instead
# of dividing it.
let LEVEL = env_or("BENCH_LEVEL", "hard")
let DROP = if LEVEL == "easy" { 4 } elif LEVEL == "medium" { 2 } else { 0 }
let N = 38 - DROP

fn fib(n: int) -> int {
    if n < 2 { return n }
    return fib(n - 1) + fib(n - 2)
}

print(fib(N))
