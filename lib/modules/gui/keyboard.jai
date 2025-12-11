import lib/modules/gui/window
import lib/modules/ds/array

#mac codes
var KEY_W = 13
var KEY_A = 0
var KEY_S = 1
var KEY_D = 2
var KEY_UP = 126
var KEY_DOWN = 125
var KEY_LEFT = 123
var KEY_RIGHT = 124
var KEY_ESC = 53
var KEY_SPACE = 49
var KEY_ENTER = 52
var KEY_RETURN = 36
var KEY_TAB = 48
var KEY_BACKSPACE = 51
var KEY_DELETE = 117#fwd
var KEY_HOME = 115
var KEY_END = 119
var KEY_PAGE_UP = 116
var KEY_PAGE_DOWN = 121

var KEY_0 = 29
var KEY_1 = 18
var KEY_2 = 19
var KEY_3 = 20
var KEY_4 = 21
var KEY_5 = 23
var KEY_6 = 22
var KEY_7 = 26
var KEY_8 = 28
var KEY_9 = 25

var KEY_MINUS = 27
var KEY_EQUAL = 24
var KEY_BACKTICK = 50
var KEY_LEFT_BRACKET = 33
var KEY_RIGHT_BRACKET = 30
var KEY_BACKSLASH = 42

var KEY_Q = 12
var KEY_E = 14
var KEY_R = 15
var KEY_T = 17
var KEY_Y = 16
var KEY_U = 32
var KEY_I = 34
var KEY_O = 31
var KEY_P = 35
var KEY_F = 3
var KEY_G = 5
var KEY_H = 4
var KEY_J = 38
var KEY_K = 40
var KEY_L = 37
var KEY_SEMICOLON = 41
var KEY_APOSTROPHE = 39
var KEY_Z = 6
var KEY_X = 7
var KEY_C = 8
var KEY_V = 9
var KEY_B = 11
var KEY_N = 45
var KEY_M = 46
var KEY_COMMA = 43
var KEY_PERIOD = 47
var KEY_SLASH = 44

var KEY_SHIFT_LEFT = 56
var KEY_SHIFT_RIGHT = 60
var KEY_CTRL_LEFT = 59
var KEY_CTRL_RIGHT = 62
var KEY_ALT_LEFT = 58 #option
var KEY_ALT_RIGHT = 61
var KEY_CMD_LEFT = 55
var KEY_CMD_RIGHT = 54
var KEY_CAPSLOCK = 57

var KEY_F1 = 122
var KEY_F2 = 120
var KEY_F3 = 99
var KEY_F4 = 118
var KEY_F5 = 96
var KEY_F6 = 97
var KEY_F7 = 98
var KEY_F8 = 100
var KEY_F9 = 101
var KEY_F10 = 109
var KEY_F11 = 103
var KEY_F12 = 111

var _prevKeys = arrayFill(256, 0)

func keyboardReset()
    _prevKeys = arrayFill(256, 0)
end

func keyboardTick(win)
    if arrayLen(_prevKeys) == 0
        keyboardReset()
    end
    var pressed = []
    var code = 0
    while code < 256
        var down = win.keyDown(code)
        if down and _prevKeys[code] == 0
            arrayPush(pressed, code)
        end
        if down
            _prevKeys[code] = 1
        else
            _prevKeys[code] = 0
        end
        code = code + 1
    end
    return pressed
end
