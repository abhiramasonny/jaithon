#: Arbitrary-precision arithmetic: `BigInt` and the one free function built
#: on it.
#:
#: `bigint.jai` stays independently importable — `from std.num.bigint import
#: BigInt` keeps working exactly as before. This facade exists for the times
#: a caller wants the family, not the file.
#:
#: ```
#: from std.num import BigInt, factorial
#: factorial(25).to_str()   # "15511210043330985984000000"
#: ```
from .bigint import BigInt, factorial

export { BigInt, factorial }
