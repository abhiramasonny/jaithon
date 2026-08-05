# expect: E0900
# `-9223372036854775808` is a whole literal, but negating it again is not
# representable, and a `const` initialiser is folded at compile time.

const AGAIN = -(-9223372036854775808)
print(AGAIN)
