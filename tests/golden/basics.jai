#: Literals, bindings, operators, and the precedence table (spec §1, §3, §4).

# --- literals ---------------------------------------------------------------
print(42)
print(1_000_000)
print(0xFF)
print(0o755)
print(0b1010)
print(3.5)
print(1e3)
print(2.5e-2)
print(true)
print(false)
print(null)
print("hello")
print('single quotes')
print(r"raw\nnot-a-newline")
print("escapes: \t| \\ \" \u{2713}")
print("""triple
line""")

# --- bindings ---------------------------------------------------------------
let name = "Jaithon"
var count = 0
count = count + 1
count += 2
print(f"{name} {count}")

const MAX = 10 * 10
print(MAX)

let x: int = 7
let y: float = 2.0
print(x)
print(y)

# --- arithmetic -------------------------------------------------------------
print(7 + 3)
print(7 - 3)
print(7 * 3)
print(7 / 2)    # int / int -> float
print(7 // 2)   # floor division
print(-7 // 2)  # floors toward negative infinity
print(7 % 3)
print(-7 % 3)  # sign follows the divisor, like Python
print(2 ** 10)
print(2 ** 3 ** 2)  # right-associative: 2 ** 9

# --- bitwise ----------------------------------------------------------------
print(0b1100 & 0b1010)
print(0b1100 | 0b1010)
print(0b1100 ^ 0b1010)
print(~5)
print(1 << 8)
print(256 >> 4)

# --- comparison and chaining ------------------------------------------------
print(1 < 2)
print(2 <= 2)
print(3 > 4)
print(1 == 1.0)
print("a" != "b")
print(1 < 2 and 2 < 3)
print(1 < 2 < 3)    # chained, b evaluated once
print(3 < 2 < 100)  # short-circuits to false

# --- logical ----------------------------------------------------------------
print(true and false)
print(true or false)
print(not true)

# --- null handling ----------------------------------------------------------
let maybe: int? = null
print(maybe ?? -1)
let present: int? = 5
print(present ?? -1)

# --- ternary ----------------------------------------------------------------
print(count > 2 ? "big" : "small")

# --- strings ----------------------------------------------------------------
print("con" + "cat")
print("ab" * 3)
print("hello".upper())
print("HELLO".lower())
print("  pad  ".strip())
print("a,b,c".split(","))
print(",".join(["x", "y", "z"]))
print("hello".len())
print("hello"[1])
print("hello"[1:3])
print("hello"[-1])
print("needle" in "a needle here")

# --- f-strings --------------------------------------------------------------
let a = 3
let b = 4
print(f"{a} + {b} = {a + b}")
print(f"{a / b:.3f}")
print(f"{{literal braces}}")
