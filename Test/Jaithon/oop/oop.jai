class Person
    var name
    var age
    
    func init(self, n, a)
        self.name = n
        self.age = a
    end
    
    func greet(self)
        print "Hello, my name is " + self.name
        print "I am " + str(self.age) + " years old"
    end
end

var p = new Person("Alice", 30)
p.greet()
print p.name
