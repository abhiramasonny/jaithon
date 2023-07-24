input firstnum 
input secnum
input thirdnum
var largenum = 0
if firstnum>secnum then do var largenum = firstnum
if secnum>firstnum then do var largenum = secnum
if largenum<thirdnum then do var largenum = thirdnum
print largenum
