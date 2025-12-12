# Test: Simple benchmark (headless)
# Expected output: 0


var sum = 0
var i = 0
while i < 500000
    sum = sum + (i % 7)
    i = i + 1
end

if sum < 0
    print sum
else
    print 0
end
