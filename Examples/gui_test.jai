import lib/modules/gui/window

func main()
    var width = 400
    var height = 300
    var win = new Window(width, height, "Jaithon GUI Test")
    
    var x = 0
    var y = 100
    var dx = 5
    
    while win.update()
        win.clear(rgb(50, 50, 50))
        
        win.fillRect(x, y, 50, 50, rgb(255, 0, 0))
        
        win.drawRect(10, 10, 100, 50, rgb(0, 255, 0))
        win.drawLine(0, 0, width, height, rgb(0, 0, 255))
        win.drawCircle(200, 150, 50, rgb(255, 255, 0))
        
        x = x + dx
        if x > width - 50 or x < 0
            dx = 0 - dx
        end
    end
end

main()
