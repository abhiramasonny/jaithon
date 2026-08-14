# expect: E0400

fn describe(values: list[int | float]) -> int { return values.len() }

let xs = [1, 2.0]
print(describe(xs))
