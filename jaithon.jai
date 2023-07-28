#NOTE THIS IS A TESTS FILE.
#comparrisions test
var nums = 1
var lessthan = nums<3
print lessthan
var greatorthan = nums>3
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
var result = math.sin(angle)
print result

#cos test
var angle = 0.8
var result = math.cos(angle)
print result

#tan test
var angle = 1.2
var result = math.tan(angle)
print result

#asin test
var value = 0.5
var result = math.asin(value)
print result

#acos test
var value = 0.8
var result = math.acos(value)
print result

#atan test
var value = 1.2
var result = math.atan(value)
print result

#sqrt test
var value = 64.0
var result = math.sqrt(value)
print result

#math opperator tests
var num = 11
var dum = 12
var ndum = num + dum
print ndum

#time test
var t = time()
print t

#not test
var c = 1
var c = c not
print c

#math test
var q = math.quad(1,2,1)
print q
var p = math.pyth(3,4)
print p
var b = math.bin.conv(10)
print b
var bb = math.bin.badd(10,10)
print bb

var jaithon = "jaithon"
print jaithon
var hello = "Hello World"
print hello

var value1 = 0
var value2 = 1
var d = value1 and value2

print d

var dd = value1 eq value2

print dd

var nice = "Are Not Equal"
if dd not then do print nice
var o = value1 or value2
print o

var x = 1
var xy = 1
var xyz = x xor xy
print xyz

var fac = 5
var fac = fac!
print fac

var exp = 5 ^ 3
print exp