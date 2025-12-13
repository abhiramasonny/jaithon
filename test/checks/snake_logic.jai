import gui/keyboard
import ds/array

class MockWindow
    var keys
    func init(self, keys)
        self.keys = keys
    end
    func getKeys(self)
        return self.keys
    end
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

func testQueueLogic()
    var dirX = 0
    var dirY = -1
    var dirQueue = []

    arrayPush(dirQueue, KEY_RIGHT)
    arrayPush(dirQueue, KEY_UP)

    var processedTurn = false
    while arrayLen(dirQueue) > 0 and not processedTurn
        var code = popFront(dirQueue)
        var ndx = dirX
        var ndy = dirY
        if code == KEY_UP
            ndx = 0
            ndy = -1
        end
        if code == KEY_RIGHT
            ndx = 1
            ndy = 0
        end

        if not (ndx == -dirX and ndy == -dirY) and not (ndx == dirX and ndy == dirY)
            dirX = ndx
            dirY = ndy
            processedTurn = true
        end
    end

    print dirX
    print dirY
    print arrayLen(dirQueue)
end

func testKeyboardTick()
    keyboardReset()
    var win = new MockWindow([])

    win.keys = [KEY_LEFT]
    var pressed = keyboardTick(win)
    print arrayLen(pressed)
    print pressed[0]

    pressed = keyboardTick(win)
    print arrayLen(pressed)

    win.keys = []
    pressed = keyboardTick(win)
    print arrayLen(pressed)

    win.keys = [KEY_LEFT]
    pressed = keyboardTick(win)
    print arrayLen(pressed)
end

testQueueLogic()
testKeyboardTick()
