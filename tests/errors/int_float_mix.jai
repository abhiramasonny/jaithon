# expect: E0411
# int widens to float, but float does not narrow to int: the fraction would be
# lost, so the truncation has to be asked for.

let a: float = 1.5
let b: int = a
print(b)
