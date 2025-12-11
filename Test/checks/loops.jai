# Test: While loops
# Expected output: 0, 1, 2, 3, 4, sum=10, countdown=5, countdown=4, countdown=3, countdown=2, countdown=1

var i = 0
while i < 5
    print i
    i = i + 1
end

var sum = 0
i = 0
while i <= 4
    sum = sum + i
    i = i + 1
end
print "sum=" + str(sum)

var count = 5
while count > 0
    print "countdown=" + str(count)
    count = count - 1
end
