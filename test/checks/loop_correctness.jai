# Test: top-level loop correctness (auto-parallelizer) + bootstrapped primitives
# Expected output: 2000, 2000, 499500, 1000, 600, 42, 30, 7, 65, 19

# 1. top-level constant-increment loop (was miscomputed as sum-of-iterator)
var n = 0
var i = 0
while i < 2000
    n = n + 1
    i = i + 1
end
print n

# 2. top-level string accumulation (was mis-tagged as numeric reduction -> 0)
var s = ""
i = 0
while i < 2000
    s = concat(s, "x")
    i = i + 1
end
print len(s)

# 3. top-level real reduction rv = rv + iter (fast closed-form path must stay correct)
var acc = 0
i = 0
while i < 1000
    acc = acc + i
    i = i + 1
end
print acc

# 4. large native array via _push (>= parallel threshold)
var arr = []
i = 0
while i < 1000
    _push(arr, i)
    i = i + 1
end
print _alen(arr)

# 5. bootstrapped dynamic array on _buf
var d = darrNew()
i = 0
while i < 600
    darrPush(d, i)
    i = i + 1
end
print darrLen(d)

# 6. bootstrapped hash table on _buf
var m = htNew()
htSet(m, "answer", 42)
print htGet(m, "answer")

# 7. general parallel map
func sq(x)
    return x * x
end
var sqs = _pmap([1, 2, 3, 4], sq)
print sqs[0] + sqs[1] + sqs[2] + sqs[3]

# 8. memory buffer primitive
var b = _buf(4)
_bset(b, 2, 7)
print _bget(b, 2)

# 9. char code primitive
print _ord("A")

# 10. gpu matmul (CPU fallback for small): [[1,2],[3,4]] x [[5,6],[7,8]] -> C[0][0]=19
var c = _gpu_matmul([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], 2, 2, 2)
print _int(c[0])
