import lib/modules/gui/window
import lib/modules/gui/keyboard
import lib/modules/core/core
import lib/modules/ds/array

func clamp(val, lo, hi)
    if val < lo
        return lo
    end
    if val > hi
        return hi
    end
    return val
end

func shiftLeft(arr)
    var i = 1
    while i < arrayLen(arr)
        arr[i - 1] = arr[i]
        i = i + 1
    end
    arrayPop(arr)
end

func popFront(arr)
    if arrayLen(arr) == 0
        return 0
    end
    var first = arr[0]
    shiftLeft(arr)
    return first
end

func randomFood(cellsX, cellsY)
    var fx = floor(rand() * cellsX)
    var fy = floor(rand() * cellsY)
    var pos = []
    arrayPush(pos, fx)
    arrayPush(pos, fy)
    return pos
end

func main()
    var win = new Window()
    var width = 400
    var height = 300
    var cell = 10
    win.init(width, height, "Snake")
    
    var cellsX = width / cell
    var cellsY = height / cell
    
    var snake = []
    arrayPush(snake, [cellsX / 2, cellsY / 2])
    var dirX = 1
    var dirY = 0
    var targetLen = 4
    var tick = 0.12
    var acc = 0.0
    var food = randomFood(cellsX, cellsY)
    var dirQueue = []
    
    keyboardReset()
    
    while true
        win.poll()
        var dt = win.getDeltaTime()
        acc = acc + dt
        
        var presses = keyboardTick(win)
        var ki = 0
        while ki < arrayLen(presses)
            arrayPush(dirQueue, presses[ki])
            ki = ki + 1
        end
        
        if win.keyDown(KEY_ESC)
            break
        end
        
        while acc >= tick
            acc = acc - tick
            
            while arrayLen(dirQueue) > 0
                var code = popFront(dirQueue)
                var ndx = dirX
                var ndy = dirY
                if code == KEY_W or code == KEY_UP
                    ndx = 0
                    ndy = -1
                end
                if code == KEY_S or code == KEY_DOWN
                    ndx = 0
                    ndy = 1
                end
                if code == KEY_A or code == KEY_LEFT
                    ndx = -1
                    ndy = 0
                end
                if code == KEY_D or code == KEY_RIGHT
                    ndx = 1
                    ndy = 0
                end
                if not (ndx == -dirX and ndy == -dirY)
                    dirX = ndx
                    dirY = ndy
                end
            end
            
            var head = snake[arrayLen(snake) - 1]
            var nx = head[0] + dirX
            var ny = head[1] + dirY
            var collided = false
            
            if nx < 0
                nx = cellsX - 1
            end
            if nx >= cellsX
                nx = 0
            end
            if ny < 0
                ny = cellsY - 1
            end
            if ny >= cellsY
                ny = 0
            end
            
            var i = 0
            while i < arrayLen(snake)
                if snake[i][0] == nx and snake[i][1] == ny
                    collided = true
                    break
                end
                i = i + 1
            end
            
            if collided
                snake = []
                arrayPush(snake, [cellsX / 2, cellsY / 2])
                dirX = 1
                dirY = 0
                targetLen = 4
                food = randomFood(cellsX, cellsY)
                acc = 0
                break
            end
            
            arrayPush(snake, [nx, ny])
            
            if nx == food[0] and ny == food[1]
                targetLen = targetLen + 1
                food = randomFood(cellsX, cellsY)
            end
            
            if arrayLen(snake) > targetLen
                shiftLeft(snake)
            end
        end
        
        win.clear(rgb(15, 15, 20))
        
        win.fillRect(food[0] * cell, food[1] * cell, cell, cell, rgb(220, 80, 80))
        
        var j = 0
        while j < arrayLen(snake)
            var s = snake[j]
            win.fillRect(s[0] * cell, s[1] * cell, cell, cell, rgb(80, 200, 120))
            j = j + 1
        end
        
        win.display()
        if not win.update()
            break
        end
    end
end

main()
