#to-do Manager
#   add <text>     - add a task
#   list           - show all tasks
#   done <id>      - mark done
#   delete <id>    - remove
#   save <path>    - write to file
#   load <path>    - read from file
#   exit           - quit

import lib/modules/hashmap
import lib/modules/vector
import lib/modules/string

func prompt()
    print "todo> "
end

func splitOnce(s, sep)
    var idx = indexOf(s, sep)
    if idx < 0
        return [s, ""]
    end
    return [substr(s, 0, idx), substr(s, idx + len(sep), len(s) - (idx + len(sep)))]
end

func trimLine(line)
    return trim(line)
end

func readLine()
    var line = ""
    input line
    return line
end

func saveFile(path, lines)
    var i = 0
    while i < _alen(lines)
        system "echo \"" + replace(_get(lines, i), "\"", "\\\"") + "\" >> " + path
        i = i + 1
    end
end

func loadFile(path)
    print "load not implemented in this environment"
end

class Task
    var id
    var text
    var done
end

func newTask(id, text)
    var t = new Task()
    t.id = id
    t.text = text
    t.done = false
    return t
end

class Todo
    var nextId
    var tasks
    
    func init(self)
        self.nextId = 1
        self.tasks = hashmap()
    end
    
    func add(self, text)
        var id = self.nextId
        self.nextId = self.nextId + 1
        self.tasks.set(num(id), newTask(id, text))
        print "Added #" + str(id) + ": " + text #weird syntax highlighting cus of the pound
    end
    
    func list(self)
        var keys = self.tasks.keys()
        var i = 0
        while i < _alen(keys)
            var k = _get(keys, i)
            var t = self.tasks.get(k)
            var status = ""
            if t.done
                status = "[x]"
            else
                status = "[ ]"
            end
            print status + " #" + str(t.id) + " " + t.text
            i = i + 1
        end
        if _alen(keys) eq 0
            print "(no tasks)"
        end
    end
    
    func markDone(self, id)
        var t = self.tasks.get(num(id))
        if t eq null
            print "No such id"
            return
        end
        t.done = true
        print "Done #" + str(id)
    end
    
    func delete(self, id)
        if self.tasks.delete(num(id))
            print "Deleted #" + str(id)
        else
            print "No such id"
        end
    end
end

func main()
    var todo = new Todo()
    while true
        prompt()
        var line = readLine()
        line = trimLine(line)
        if len(line) eq 0

        else
            if line eq "exit"
                return
            else
                if line eq "list"
                    todo.list()
                else
                    if startsWith(line, "add ")
                        todo.add(substr(line, 4, len(line) - 4))
                    else
                        if startsWith(line, "done ")
                            todo.markDone(substr(line, 5, len(line) - 5))
                        else
                            if startsWith(line, "delete ")
                                todo.delete(substr(line, 7, len(line) - 7))
                            else
                                print "Commands: add <text>, list, done <id>, delete <id>, exit"
                            end
                        end
                    end
                end
            end
        end
    end
end

main()
