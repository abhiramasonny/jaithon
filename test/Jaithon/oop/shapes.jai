class Point
    var x
    var y

    func init(self, x, y)
        self.x = x
        self.y = y
    end

    func show(self)
        print "Point(" + str(self.x) + ", " + str(self.y) + ")"
    end

    func distance(self, other)
        var dx = self.x - other.x
        var dy = self.y - other.y
        return _sqrt(dx * dx + dy * dy)
    end
end

class Shape
    var name

    func init(self, name)
        self.name = name
    end

    func area(self)
        return 0
    end

    func describe(self)
        print "Shape: " + self.name
    end
end

class Rectangle extends Shape
    var width
    var height

    func init(self, w, h)
        self.name = "Rectangle"
        self.width = w
        self.height = h
    end

    func area(self)
        return self.width * self.height
    end

    func perimeter(self)
        return 2 * (self.width + self.height)
    end

    func describe(self)
        print "Rectangle: " + str(self.width) + " x " + str(self.height)
        print "Area: " + str(self.area())
        print "Perimeter: " + str(self.perimeter())
    end
end

class Circle extends Shape
    var radius

    func init(self, r)
        self.name = "Circle"
        self.radius = r
    end

    func area(self)
        return 3.14159 * self.radius * self.radius
    end

    func circumference(self)
        return 2 * 3.14159 * self.radius
    end

    func describe(self)
        print "Circle with radius: " + str(self.radius)
        print "Area: " + str(self.area())
        print "Circumference: " + str(self.circumference())
    end
end

print "=== Point Tests ==="
var p1 = new Point(0, 0)
var p2 = new Point(3, 4)
p1.show()
p2.show()
print "Distance between p1 and p2:"
print p1.distance(p2)

print ""
print "=== Rectangle Tests ==="
var rect = new Rectangle(5, 3)
rect.describe()

print ""
print "=== Circle Tests ==="
var circle = new Circle(7)
circle.describe()
