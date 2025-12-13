# Logical short-circuit behavior
# Expected output:
# 0
# 0
# 1
# 2

var hits = 0

func bump()
    hits = hits + 1
    return true
end

if false and bump()
    print "should not print"
end
print hits

if true or bump()
    print "skip second operand"
end
print hits

if false or bump()
    print "bumped"
end
print hits

if bump() and bump()
    print "both evaluated"
end
print hits
