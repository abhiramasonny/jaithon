var msg = "Think of a number between 0 and 100. Tell me lower for lower, and higher for higher. I will guess it. If i do guess it, (or get close) then type yes"
print msg
var low = 0
var high = 100
var mid = 0
var win = "HAHAHA I won!! Your number is: "
var h = "higher"
var l = "lower"
var y = "yes"
while low<high
    var mid = low + high
    var mid = mid / 2
    var mid = math.round(mid)
    print mid
    input HigherorLower, string
    if compare HigherorLower, h then do var low = mid + 1
    if compare HigherorLower, l then do var high = mid - 1
    if compare HigherorLower, y then do print win
    if compare HigherorLower, y then do print mid
    if compare HigherorLower, y then do break
end
