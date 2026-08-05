# expect: E0400
# The one thing fusion breaks, kept as a diagnostic rather than a surprise.
# `[1, 2.0]` used to infer `list[int | float]` and would have been accepted
# here; it now infers `list[float]`, and lists are invariant, so the annotation
# has to say what the literal actually is.

fn describe(values: list[int | float]) -> int { return values.len() }

let xs = [1, 2.0]
print(describe(xs))
