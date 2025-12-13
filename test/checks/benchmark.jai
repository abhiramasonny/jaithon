# Test: Simple benchmark (headless)
# Expected output: 14999995

var sum = 0
var i = 0
while i < 5000000
    sum = sum + (i % 7)
    i = i + 1
end

print sum

