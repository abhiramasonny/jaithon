# expect: E0400
# Fusion converts the elements of `[1, 2.0]` and makes it a `list[float]`. It
# does not make `int` a subtype of `float`, so the result is still not a
# `list[int]`: lists are invariant, and there is no place to hang a per-element
# conversion on a list that already exists.

fn total(values: list[int]) -> int { return values.len() }

let xs = [1, 2.0]
print(total(xs))
