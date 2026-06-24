# Test: new language features (for/try/fstring/dict/compound/continue)
# Expected output: hello world, 5, boom, finally, 60, abc, 1, 2, 50, 19.5, 1, 9, 1, hello-world-foo

# f-strings
var name = "world"
print f"hello {name}"

# break in while
var i = 0
while i < 10 do
  i += 1
  if i == 5 then
    break
  end
end
print i

# try/except/finally
try
  raise "boom"
except err do
  print err
finally do
  print "finally"
end

# for x in array
var arr = [10, 20, 30]
var total = 0
for item in arr do
  total += item
end
print total

# for x in string
var letters = ""
for c in "abc" do
  letters += c
end
print letters

# dict literal subscript
var d = {"a": 1, "b": 2}
print d["a"]
print d["b"]

# continue
var sum = 0
var j = 0
while j < 10 do
  j += 1
  if j == 5 then
    continue
  end
  sum += j
end
print sum

# compound assignment
var y = 10
y += 5
y -= 2
y *= 3
y /= 2
print y

# sort
var nums = [5, 3, 8, 1, 9, 2, 7, 4, 6]
sort(nums)
print nums[0]
print nums[8]
print isSorted(nums)

# split/join
var parts = split("hello world foo", " ")
print join(parts, "-")
