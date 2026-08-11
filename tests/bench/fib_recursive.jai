#: Deep recursion — measures call-frame setup and teardown.
fn fib(n: int) -> int {
    if n < 2 { return n }
    return fib(n - 1) + fib(n - 2)
}

fn main() -> int {
    print(fib(300))
    return 0
}
