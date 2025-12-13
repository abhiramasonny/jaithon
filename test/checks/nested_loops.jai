# Test: Nested loops and break
# Expected output: 0-0, 0-1, 0-2, 1-0, 1-1, 1-2, done, 0, 1, 2, stopped

var i = 0
while i < 2
    var j = 0
    while j < 3
        print str(i) + "-" + str(j)
        j = j + 1
    end
    i = i + 1
end
print "done"

var k = 0
while k < 10
    if k == 3
        print "stopped"
        break
    end
    print k
    k = k + 1
end
