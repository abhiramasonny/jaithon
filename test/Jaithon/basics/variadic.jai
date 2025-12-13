func sum(*args)
    var total = 0
    var i = 0
    while i < _alen(args)
        total = total + args[i]
        i = i + 1
    end
    return total
end

func average(*nums)
    var total = 0
    var i = 0
    while i < _alen(nums)
        total = total + nums[i]
        i = i + 1
    end
    return total / _alen(nums)
end

func max(*values)
    if _alen(values) == 0
        return 0
    end
    var m = values[0]
    var i = 1
    while i < _alen(values)
        if values[i] > m
            m = values[i]
        end
        i = i + 1
    end
    return m
end

func min(*values)
    if _alen(values) == 0
        return 0
    end
    var m = values[0]
    var i = 1
    while i < _alen(values)
        if values[i] < m
            m = values[i]
        end
        i = i + 1
    end
    return m
end

func printAll(*items)
    print "Printing all items:"
    var i = 0
    while i < _alen(items)
        print items[i]
        i = i + 1
    end
end

print "=== Sum Tests ==="
print "sum(1, 2, 3):"
print sum(1, 2, 3)

print "sum(10, 20, 30, 40, 50):"
print sum(10, 20, 30, 40, 50)

print ""
print "=== Average Test ==="
print "average(10, 20, 30):"
print average(10, 20, 30)

print ""
print "=== Max/Min Tests ==="
print "max(5, 2, 8, 1, 9, 3):"
print max(5, 2, 8, 1, 9, 3)

print "min(5, 2, 8, 1, 9, 3):"
print min(5, 2, 8, 1, 9, 3)

print ""
print "=== Print All Test ==="
printAll("hello", 42, 3.14, "world")
