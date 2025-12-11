#Move a square with the keyboard WASD

import lib/modules/gui/window

func main()
    var win = new Window()
    win.init(400, 300, "Keyboard Move")
    
    var x = 200
    var y = 150
    var size = 20
    var speed = 200 #pps
    
    while true
        win.poll()
        var dt = win.getDeltaTime()
        
        if win.keyDown(13) or win.keyDown(126) # W
            y = y - (speed * dt)
        end
        if win.keyDown(1) or win.keyDown(125)  # S
            y = y + (speed * dt)
        end
        if win.keyDown(0) or win.keyDown(123)  # A
            x = x - (speed * dt)
        end
        if win.keyDown(2) or win.keyDown(124)  # D
            x = x + (speed * dt)
        end
        
        if x < 0
            x = 0
        end
        if y < 0
            y = 0
        end
        if x > win.width - size
            x = win.width - size
        end
        if y > win.height - size
            y = win.height - size
        end
        
        win.clear(rgb(20, 20, 30))
        win.fillRect(x, y, size, size, rgb(50, 200, 255))
        
        win.display()
        if not win.update()
            break
        end
    end
end

main()
