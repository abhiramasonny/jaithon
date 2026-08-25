# expect: E0406

#: `@` is a binary operator, so a newline after one continues the line -- the
#: same rule `and`, `or`, `in` and `is` live under. It was left out of the
#: lexer's table when the operator was added, and the tell is WHICH error this
#: file reports: E0102 "expected an expression, found a newline" means the
#: newline ended the statement, E0406 "operator `@` does not apply" means the
#: line carried on and the checker got both operands. Only the second is right.
#:
#: There is no runnable test for `@` because nothing implements it: the checker
#: answers `<unknown>` for every operand pair (see tests/checker's operator
#: table), so the furthest a program can get is exactly this diagnostic.
let left = 3
let right = 4
let product = left @
    right
print(product)
