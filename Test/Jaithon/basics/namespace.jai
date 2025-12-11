namespace Math
    var PI = 3.14159
    
    func area(r)
        return PI * r * r
    end
    
    func doubleArea(r)
        return 2 * area(r)
    end
end

print Math.PI
print Math.area(10)
print Math.doubleArea(10)

Math.PI = 3
print Math.PI
print Math.area(10)
