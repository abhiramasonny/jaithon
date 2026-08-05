# expect: E0303
# `const` initialisers must be evaluable at compile time.

fn compute() -> int { return 4 }
const SIZE = compute()
print(SIZE)
