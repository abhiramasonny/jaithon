import lib/modules/vector

func main()
    var v = vector()
    v.push(1)
    v.push(2)
    v.push(3)
    print "len=" + str(v.len())
    print v.get(0)
    print v.get(1)
    v.set(1, 42)
    print v.get(1)
    print v.pop()
end

main()
