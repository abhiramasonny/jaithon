# Test: If-else conditionals
# Expected output: greater, equal, in_else, positive, zero_or_negative, zero_or_negative

var x = 10
var y = 5

if x > y
    print "greater"
end

if x == 10
    print "equal"
end

if x < y
    print "in_if"
else
    print "in_else"
end

var n = 5
if n > 0
    print "positive"
else
    print "zero_or_negative"
end

n = 0
if n > 0
    print "positive"
else
    print "zero_or_negative"
end

n = -3
if n > 0
    print "positive"
else
    print "zero_or_negative"
end
