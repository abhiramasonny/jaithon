# expect: E0301
# Reassigning a `let` binding is a compile error.

let x = 1
x = 2
print(x)
