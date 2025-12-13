# Test: Classes and objects
# Expected output: 0, 42, Alice, 30, area=15, perimeter=16

class Counter
    var count

    func init(self)
        self.count = 0
    end

    func get(self)
        return self.count
    end

    func set(self, val)
        self.count = val
    end
end

var c = new Counter()
print c.get()
c.set(42)
print c.get()

class Person
    var name
    var age

    func init(self, n, a)
        self.name = n
        self.age = a
    end

    func getName(self)
        return self.name
    end

    func getAge(self)
        return self.age
    end
end

var p = new Person("Alice", 30)
print p.getName()
print p.getAge()

class Rectangle
    var width
    var height

    func init(self, w, h)
        self.width = w
        self.height = h
    end

    func area(self)
        return self.width * self.height
    end

    func perimeter(self)
        return 2 * (self.width + self.height)
    end
end

var rect = new Rectangle(5, 3)
print "area=" + str(rect.area())
print "perimeter=" + str(rect.perimeter())
