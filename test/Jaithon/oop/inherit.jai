class Animal
    var name
    func init(self, name)
        self.name = name
    end
    func speak(self)
        print self.name + " makes a sound"
    end
end

class Dog extends Animal
    var breed
    func init(self, name, breed)
        self.name = name
        self.breed = breed
    end
    func speak(self)
        print self.name + " barks!"
    end
    func info(self)
        print "Breed: " + self.breed
    end
end

var a = new Animal("Generic")
a.speak()

var d = new Dog("Buddy", "Golden Retriever")
d.speak()
d.info()
print d.name
