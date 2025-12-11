input N
var largest = 0
while N > 0
    input num
    if num > largest
        largest = num
    end
    N = N - 1
    if N eq 0
        print largest
        break
    end
end
