input N
var largest = 0
while N>0 do
    input num
    if num > largest then do var largest = num
    var N = N -1
    if N eq 0 then do print largest
    if N eq 0 then do break
loop