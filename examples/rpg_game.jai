import ds/hashmap
import core/string
import core/core

namespace UI
    func clear()
    end

    func header(title)
        print "========================================"
        print "   " + title
        print "========================================"
    end

    func prompt(msg)
        return _input(msg + " ")
    end

    func log(msg)
        print "[LOG] " + msg
    end
end

class Player
    var name
    var hp
    var maxHp
    var inventory
    var currentRoom

    func init(self, name)
        self.name = name
        self.hp = 100
        self.maxHp = 100
        self.inventory = []
        self.currentRoom = null
    end

    func status(self)
        print "--- Status ---"
        print "Name: " + self.name
        print "HP: " + self.hp + "/" + self.maxHp
        print "Inventory: " + str(self.inventory)
        print "--------------"
    end
end

class Room
    var name
    var description
    var exits

    func init(self, name, description)
        self.name = name
        self.description = description
        self.exits = new HashMap()
    end

    func addExit(self, direction, room)
        self.exits.set(direction, room)
    end

    func getExit(self, direction)
        return self.exits.get(direction)
    end

    func look(self)
        print "--- " + self.name + " ---"
        print self.description
    end
end

namespace Game
    var running = true
    var player = null
    var rooms = []

    func init()
        var startRoom = new Room("Start Room", "You are in a small, dimly lit room.")
        var hall = new Room("Hallway", "A long hallway with stone walls.")
        var armory = new Room("Armory", "Racks of rusty weapons line the walls.")

        startRoom.addExit("north", hall)
        hall.addExit("south", startRoom)
        hall.addExit("east", armory)
        armory.addExit("west", hall)

        rooms = [startRoom, hall, armory]

        UI.clear()
        UI.header("CHARACTER CREATION")
        var name = UI.prompt("Enter your name:")
        if name == "" then name = "Hero" end

        player = new Player(name)
        player.currentRoom = startRoom

        print "Welcome, " + player.name + "!"
        _system("sleep 1")
    end

    func quit()
        running = false
        print "Thanks for playing!"
    end

    func help()
        print "Commands: look, status, go [dir], quit, help"
    end

    func process(cmd)
        if cmd == "quit"
            quit()
            return null
        end

        if cmd == "help"
            help()
            return null
        end

        if cmd == "status"
            player.status()
            return null
        end

        if cmd == "look"
            player.currentRoom.look()
            return null
        end

        if startsWith(cmd, "go ")
            var dir = substr(cmd, 3, len(cmd) - 3)
            var nextRoom = player.currentRoom.getExit(dir)
            if nextRoom != null
                player.currentRoom = nextRoom
                player.currentRoom.look()
            else
                print "You can't go that way."
            end
            return null
        end

        print "Unknown command."
    end
end

Game.init()
UI.clear()
Game.player.currentRoom.look()

while Game.running
    var cmd = UI.prompt("What do you want to do?")
    if cmd == ""
    else
        Game.process(trim(cmd))
    end
end
