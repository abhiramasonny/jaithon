# Test: GUI module smoke (headless)
# Expected output: 1, 1, 1

import lib/modules/gui/window

print rgb(0, 0, 0) == 0
print rgb(255, 0, 0) == 255 * 65536
print rgb(1, 2, 3) == (1 * 65536) + (2 * 256) + 3
