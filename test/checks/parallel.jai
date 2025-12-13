# Test: Parallel execution and hardware acceleration
# Expected output: GPU_MODULO:1.49999e+06, CPU_FORMULA:4.9995e+07, PARALLEL_LARGE:999999, SERIAL_COMPLEX:6, SERIAL_SMALL:45, ALL_CORRECT

# ============================================================================
# TEST 1: GPU Acceleration (Metal)
# Pattern: sum = sum + (i % constant)
# Triggers: GPU compute shader when iterations >= 5000
# ============================================================================
var gpu_sum = 0
var i = 0
while i < 500000
    gpu_sum = gpu_sum + (i % 7)
    i = i + 1
end
print "GPU_MODULO:" + str(gpu_sum)

var formula_sum = 0
var j = 0
while j < 10000
    formula_sum = formula_sum + j
    j = j + 1
end
print "CPU_FORMULA:" + str(formula_sum)

var parallel_sum = 0
var k = 0
while k < 1000000
    parallel_sum = parallel_sum + (k % 3)
    k = k + 1
end
print "PARALLEL_LARGE:" + str(parallel_sum)

func isValid(x)
    return x < 4
end

var serial_complex = 0
var m = 0
while m < 10 and isValid(m)
    serial_complex = serial_complex + m
    m = m + 1
end
print "SERIAL_COMPLEX:" + str(serial_complex)

var small_sum = 0
var n = 0
while n < 10
    small_sum = small_sum + n
    n = n + 1
end
print "SERIAL_SMALL:" + str(small_sum)

var all_correct = 1

if gpu_sum != 1499994
    all_correct = 0
end

if formula_sum != 49995000
    all_correct = 0
end

if parallel_sum != 999999
    all_correct = 0
end

if serial_complex != 6
    all_correct = 0
end

if small_sum != 45
    all_correct = 0
end

if all_correct == 1
    print "ALL_CORRECT"
else
    print "SOME_FAILED"
end
