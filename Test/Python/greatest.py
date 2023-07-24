firstnum = float(input("Enter a value for firstnum: "))
secnum = float(input("Enter a value for secnum: "))
thirdnum = float(input("Enter a value for thirdnum: "))
largenum = firstnum
if secnum>firstnum: 
    largenum = secnum
if largenum<thirdnum: 
    largenum = thirdnum
print(largenum)
