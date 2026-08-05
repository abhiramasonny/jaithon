# expect: E0401
# There is no truthiness in Jaithon: a condition must be a bool.

let items = [1, 2, 3]
if items {
    print("non-empty")
}
