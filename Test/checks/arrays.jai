# Test: Arrays
# Expected output: 5, 10, 30, 50, 5, 60, 0, 2, hello, world

#creation and access
var arr = [10, 20, 30, 40, 50]
print _alen(arr)
print arr[0]
print arr[2]
print arr[4]

arr[2] = 999
print _alen(arr)

_push(arr, 60)
print arr[5]

var empty = []
print _alen(empty)

var strs = ["hello", "world"]
print _alen(strs)
print strs[0]
print strs[1]
