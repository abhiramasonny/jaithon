fn deep(n: int) -> int {
    if n <= 0 { return 0 }
    return 1 + deep(n - 1)
}

fn main() -> int {
    for _r in 0..64 { deep(1) }
    try {
        print(deep(1100))
    } catch _e: RuntimeError {
        print("RuntimeError")
    }
    return 0
}
