print "=== Array Literal Tests ==="
var arr = [10, 20, 30, 40, 50]
print "Array created: [10, 20, 30, 40, 50]"
print "Length:"
print _alen(arr)

print "Element at index 0:"
print arr[0]
print "Element at index 2:"
print arr[2]
print "Element at index 4:"
print arr[4]

print ""
print "=== Array Modification ==="
arr[2] = 999
print "After arr[2] = 999, element at 2:"
print arr[2]

print ""
print "=== Push and Pop ==="
_push(arr, 60)
_push(arr, 70)
print "After pushing 60, 70. Length:"
print _alen(arr)

print "Popped:"
print _pop(arr)
print "Length after pop:"
print _alen(arr)

print ""
print "=== Empty Array ==="
var empty = []
print "Empty array length:"
print _alen(empty)
_push(empty, "hello")
_push(empty, "world")
print "After pushing 'hello' and 'world':"
print _get(empty, 0)
print _get(empty, 1)

print ""
print "=== Nested Arrays ==="
var matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print "3x3 matrix created"
print "matrix[0][0]:"
print matrix[0][0]
print "matrix[1][1]:"
print matrix[1][1]
print "matrix[2][2]:"
print matrix[2][2]

print ""
print "=== Array Iteration ==="
var nums = [1, 2, 3, 4, 5]
var sum = 0
var i = 0
while i < _alen(nums)
    sum = sum + nums[i]
    i = i + 1
end
print "Sum of [1,2,3,4,5]:"
print sum

print ""
print "=== Array of Strings ==="
var names = ["Alice", "Bob", "Charlie"]
i = 0
while i < _alen(names)
    print "Hello, " + names[i] + "!"
    i = i + 1
end
