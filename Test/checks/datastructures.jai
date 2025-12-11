# Test: Data structures (stack, queue, vector)
# Expected output: 1, 2, 20, 20, 10, null, 1, 2, a, a, b, null, 1, 2, 5, 5, 1, 1

#Stack
var s = stack()
print s.isEmpty()
s.push(10)
s.push(20)
print s.size()
print s.peek()
print s.pop()
print s.pop()
print s.pop()
print s.isEmpty()

#queue
var q = queue()
q.enqueue("a")
q.enqueue("b")
print q.size()
print q.peek()
print q.dequeue()
print q.dequeue()
print q.dequeue()
print q.isEmpty()

#vector
var v = vector()
v.push(1)
v.push(2)
v.set(1, 5)
print v.len()
print v.get(1)
print v.pop()
print v.len()
var arr = v.toArray()
print arr[0]
