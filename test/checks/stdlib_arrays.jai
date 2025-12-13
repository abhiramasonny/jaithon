# Stdlib array helpers
# Expected output:
# 28
# 7
# 5
# 3
# 7
# 3
# 4
# 6
# 7
# 9
# 10
# 4
# 1
# 6
# 6
# 14
# 2
# 4
# 6
# 25
# 1
# 0
# 4
# -1

import ds/array

func doubleVal(x)
    return x * 2
end

func even(x)
    return x % 2 == 0
end

func add(acc, x)
    return acc + x
end

var filled = arrayFill(4, 7)
print arraySum(filled)
print arrayMax(filled)

var rng = arrayRange(3, 8)
print arrayLen(rng)
print rng[0]
print rng[4]

var slice = arraySlice(rng, 1, 4)
print arrayLen(slice)
print slice[0]
print slice[2]

var concat = arrayConcat(rng, [9, 10])
print arrayLen(concat)
print concat[5]
print concat[6]

var reversed = arrayReverse([1, 2, 3, 4])
print reversed[0]
print reversed[3]

var mapped = arrayMap(rng, doubleVal)
print mapped[0]
print mapped[4]

var filtered = arrayFilter(rng, even)
print arrayLen(filtered)
print filtered[0]
print filtered[1]

print arrayReduce(rng, 0, add)
print arrayContains(rng, 6)
print arrayContains(rng, 2)
print arrayIndexOf(rng, 7)
print arrayIndexOf(rng, 42)
