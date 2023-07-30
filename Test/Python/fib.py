n1,n2,count = 0,1,0
msg = "Enter how many terms: "
print(msg)
nterms = float(input("Enter a value for nterms: "))
while count < nterms:
    print(n1)
    nth = n1 + n2
    n1 = n2
    n2 = nth
    count += 1
