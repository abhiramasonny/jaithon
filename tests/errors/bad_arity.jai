# expect: E0602
# Arity is checked at statically known call sites.

fn add(a: int, b: int) -> int { return a + b }
print(add(1))
