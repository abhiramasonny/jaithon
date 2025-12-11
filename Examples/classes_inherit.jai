#inheritance and method override demo

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
    func init(self, name)
        self.name = name
    end
    func speak(self)
        print self.name + " barks"
    end
end

func main()
    var a = new Animal("Creature")
    var d = new Dog("Rex")
    a.speak()
    d.speak()
end

main()
