var msg = "Think of a number between 0 and 100. Tell me 0 for lower, and 1 for higher. I will guess it. If i do guess it, (or get close) then type 2"
print msg
var low = 0
var high = 100
var mid = 0
var win = "HAHAHA I won!! Your number is: "
while low<high do
    var mid = low + high
    var mid = mid / 2
    var mid = math.round(mid)
    print mid
    input HigherorLower
    if HigherorLower eq 1 then do var low = mid + 1
    if HigherorLower eq 0 then do var high = mid - 1
    if HigherorLower eq 2 then do print win
    if HigherorLower eq 2 then do print mid
    if HigherorLower eq 2 then do break
loop
