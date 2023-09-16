#NOTE THIS IS A TESTS FILE.
#comparrisions test
var nums = 1
var lessthan = nums<3
print lessthan
var greatorthan = nums>3
print greatorthan
if lessthan then do print 1

#sin test
var angle = 0.5
var result = math.sin(angle, deg)
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
var t = time
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
var tt = math.root(3,125)
print tt

var ax = 4
var bx = 2
var ay = 6
var by = 9
var value = math.dist(ax,ay,bx,by)
print value

var thing = 140.532
var thing = math.round(thing)
print thing
var s1 = "hii"
var s2 = "hi"
var s3 = compare s1, s2
print s3
var b = 10
#while b > 1 do
#    print b 
#    var b = b - 1
#loop
#input h, string
#print h
#var h2 = "*"
#if compare h, h2 then do print h

var a = 101
var b = a % 2
print b
var r = math.rand()
var r  = r % 100
print r
var test = "Test/test.txt"
write test, hello
var abc = read test
print abc