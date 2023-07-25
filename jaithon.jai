#NOTE THIS IS A TESTS FILE.
#comparrisions test
var nums = 1
var lessthan = nums<3 #output should be 1
print lessthan
var greatorthan = nums>3 # output should be 0
print greatorthan
if lessthan then do print 1
#array test
array a = [5]
print a
a.add(0, 120.012)
a.add(2, 3)
a.add(4,120.013)
print a
a.del(4)
print a
a.sort()
print a

#sin test
var angle = 0.5
var result = sin(angle)
print result
#cos test
var angle = 0.8
var result = cos(angle)
print result

#tan test
var angle = 1.2
var result = tan(angle)
print result

#asin test
var value = 0.5
var result = asin(value)
print result

#acos test
var value = 0.8
var result = acos(value)
print result

#atan test
var value = 1.2
var result = atan(value)
print result

#import calculator
#input test
#input x
#print x

#sqrt test
var value = 64.0
var result = sqrt(value)
print result
#math opperator tests
var num = 11
var dum = 12
var ndum = num + dum
print ndum

var t = time()
print t

#import calculator

#not test
var c = 1
var c = c not
print c

#quadratic test
var q = quad(1,2,1)
print q