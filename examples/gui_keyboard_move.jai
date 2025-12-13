import lib/modules/gui/window
import lib/modules/gui/keyboard
import lib/modules/ds/array

public static void main()
    var win = new Window()
    win.init(400, 300, "Keyboard Move (WASD / Arrows)")

    float x = 200
    float y = 150
    int size = 20
    float speed = 200

    keyboardReset()

    while true
        win.poll()
        float dt = win.getDeltaTime()

        var keys = win.getKeys()
        var up = arrayContains(keys, KEY_W) or arrayContains(keys, KEY_UP)
        var down = arrayContains(keys, KEY_S) or arrayContains(keys, KEY_DOWN)
        var left = arrayContains(keys, KEY_A) or arrayContains(keys, KEY_LEFT)
        var right = arrayContains(keys, KEY_D) or arrayContains(keys, KEY_RIGHT)

        if up
            y = y - (speed * dt)
        end
        if down
            y = y + (speed * dt)
        end
        if left
            x = x - (speed * dt)
        end
        if right
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
