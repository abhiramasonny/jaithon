# expect: E0400
fn total(values: list[int]) -> int { return values.len() }

let xs = [1, 2.0]
print(total(xs))
