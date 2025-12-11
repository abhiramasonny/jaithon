# Test: Namespaces
# Expected output: 78.5398, 31.4159, 200, 300

namespace Circle
    var PI = 3.14159
    
    func area(r)
        return PI * r * r
    end
    
    func circumference(r)
        return 2 * PI * r
    end
end

print Circle.area(5)
print Circle.circumference(5)

namespace Utils
    func double(x)
        return x * 2
    end
    
    func triple(x)
        return x * 3
    end
end

print Utils.double(100)
print Utils.triple(100)
