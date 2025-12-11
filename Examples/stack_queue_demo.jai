import lib/modules/stack
import lib/modules/queue

func main()
    print "Stack demo:"
    var s = stack()
    s.push(10)
    s.push(20)
    s.push(30)
    print s.pop()
    print s.peek()
    print s.size()
    
    print "Queue demo:"
    var q = queue()
    q.enqueue("a")
    q.enqueue("b")
    q.enqueue("c")
    print q.dequeue()
    print q.peek()
    print q.size()
end

main()
