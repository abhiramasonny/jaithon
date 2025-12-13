import gui/window

public static void main()
    int width = 400
    int height = 300
    var win = new Window(width, height, "Jaithon GPU Test")
    float x = 0
    float y = 100
    float speed = 200
    int direction = 1
    int frameCount = 0

    while true
        win.poll()
        float dt = win.getDeltaTime()

        win.clear(rgb(30, 30, 40))
        win.fillRect(x, y, 50, 50, rgb(255, 100, 100))

        x = x + (speed * dt * direction)

        if x > width - 50
            x = width - 50
            direction = 0 - 1
        end
        if x < 0
            x = 0
            direction = 1
        end

        frameCount = frameCount + 1
        if frameCount == 60
            print("FPS: " + win.getFPS())
            frameCount = 0
        end

        win.display()
        if not win.update()
            break
        end
    end
end

main()
