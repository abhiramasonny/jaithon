# expect: E0402
# `Self` names the implementing type and has no meaning outside a class, trait
# or enum body (spec §7.2).

fn widen(value: Self) -> int { return 1 }

print(widen(1))
