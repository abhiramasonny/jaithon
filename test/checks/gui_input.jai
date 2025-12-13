# Test: GUI input helpers (headless)
# Expected output: 2, 0, 0, 0, 0

import gui/window

var pos = mousePos()
print _alen(pos)
print pos[0]
print pos[1]
print mouseDown(0)
print keyDown(0)
