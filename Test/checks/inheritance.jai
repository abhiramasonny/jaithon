# Test: Class inheritance
# Expected output: animal_speaks, dog_barks, Buddy, cat_meows, Whiskers

class Animal
    var name
    
    func init(self, n)
        self.name = n
    end
    
    func speak(self)
        print "animal_speaks"
    end
    
    func getName(self)
        return self.name
    end
end

class Dog extends Animal
    func init(self, n)
        self.name = n
    end
    
    func speak(self)
        print "dog_barks"
    end
end

class Cat extends Animal
    func init(self, n)
        self.name = n
    end
    
    func speak(self)
        print "cat_meows"
    end
end

var a = new Animal("Generic")
a.speak()

var d = new Dog("Buddy")
d.speak()
print d.getName()

var c = new Cat("Whiskers")
c.speak()
print c.getName()
