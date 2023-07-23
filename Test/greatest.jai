input firstnum 
input secnum
input thirdnum
var largenum = 0
if firstnum>secnum do var largenum = firstnum
if secnum>firstnum do var largenum = secnum
if largenum<thirdnum do var largenum = thirdnum
print largenum
