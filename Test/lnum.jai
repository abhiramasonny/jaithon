input N
var N = N/2
var N = math.round(N)
var largest = 0
while N>0 do
    input num1
    input num2
    if num1>largest then do var largest = num1
    if num2> largest then do var largest = num2
    var N = N -1
    if N eq 0 then do print largest
loop