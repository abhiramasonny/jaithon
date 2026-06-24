import std

# Benchmark: GPU matmul (Metal, _gpu_matmul) vs pure-Jaithon CPU matmul.
# Proves the GPU path is correct (within float32 precision) and faster on
# large matrices. The C builtin auto-falls back to CPU below ~100k FLOPs.

func cpuMatmul(a, b, M, K, N)
    var c = []
    var r = 0
    while r < M
        var col = 0
        while col < N
            var sum = 0.0
            var k = 0
            while k < K
                sum = sum + a[r * K + k] * b[k * N + col]
                k = k + 1
            end
            _push(c, sum)
            col = col + 1
        end
        r = r + 1
    end
    return c
end

func randFlat(n)
    var a = []
    var i = 0
    while i < n
        _push(a, _rand() * 2.0 - 1.0)
        i = i + 1
    end
    return a
end

func absf(x)
    if x < 0.0
        return 0.0 - x
    end
    return x
end

print "================================================="
print "  GPU matmul benchmark (Metal vs Jaithon CPU)"
print "================================================="
print concat("  GPU available: ", _str(_gpu_available()))
print ""

var sizes = [64, 128, 256]
var si = 0
while si < _alen(sizes)
    var n = sizes[si]
    var A = randFlat(n * n)
    var B = randFlat(n * n)

    var t0 = _time()
    var gc = _gpu_matmul(A, B, n, n, n)
    var t1 = _time()

    var t2 = _time()
    var cc = cpuMatmul(A, B, n, n, n)
    var t3 = _time()

    # max abs diff (float32 GPU vs float64 CPU) — should be tiny
    var maxdiff = 0.0
    var i = 0
    while i < n * n
        var dd = absf(gc[i] - cc[i])
        if dd > maxdiff
            maxdiff = dd
        end
        i = i + 1
    end

    var gpuMs = _int((t1 - t0) * 1000.0)
    var cpuMs = _int((t3 - t2) * 1000.0)
    var line = concat(_str(n), concat("x", _str(n)))
    line = concat(line, concat(" matmul   GPU=", concat(_str(gpuMs), "ms")))
    line = concat(line, concat("   CPU=", concat(_str(cpuMs), "ms")))
    line = concat(line, concat("   maxdiff=", _str(maxdiff)))
    print line
    si = si + 1
end

print ""
print "  (GPU=float32, CPU=float64 -> small maxdiff is expected & correct)"
