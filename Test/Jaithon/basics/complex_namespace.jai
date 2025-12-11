namespace Geometry
    var PI = 3.14159
    
    func circleArea(r)
        return PI * r * r
    end
    
    func circleCircumference(r)
        return 2 * PI * r
    end
end

namespace Physics
    var G = 9.81
    
    func potentialEnergy(m, h)
        return m * G * h
    end
    
    func kineticEnergy(m, v)
        return 0.5 * m * v * v
    end
end

print Geometry.circleArea(5)
print Geometry.circleCircumference(5)

print Physics.potentialEnergy(10, 5)
print Physics.kineticEnergy(10, 20)

Physics.G = 10
print Physics.potentialEnergy(10, 5)

namespace Utils
    var count = 0
    
    func increment()
        count = count + 1
        return count
    end
    
    func getCount()
        return count
    end
end

print Utils.increment()
print Utils.increment()
print Utils.getCount()
