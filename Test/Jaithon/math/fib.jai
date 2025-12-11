var n1 = 0
var n2 = 1
var count = 0

input nterms

while count < nterms
    print n1
    var nth = n1 + n2
    var n1 = n2
    var n2 = nth
    var count = count + 1
end

