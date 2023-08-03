var t = 0
var cp = 0
var i = 0
while 1 do
    var x = math.rand.uniform()
    var y = math.rand.uniform()
    var t = t + 1
    var d = x^2 + y^2
    if d < 1 then do var cp = cp + 1
    if d eq 1 then do var cp = cp + 1
    var pi = 4 * cp / t
    print pi
    var i = i + 1
loop