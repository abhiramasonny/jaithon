input firstnum 
input secnum
input thirdnum
var largenum = firstnum
if secnum>firstnum then do var largenum = secnum
if largenum<thirdnum then do var largenum = thirdnum
print largenum
