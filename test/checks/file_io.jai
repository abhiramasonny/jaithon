# File IO smoke test
# Expected output:
# 1
# file
# hello
# world
# again
# 1

import lib/modules/core/io

var path = "/tmp/jaithon_io_test.txt"

print writeFile(path, "hello\nworld")

var f = open(path, "a")
print type(f)
write(f, "\nagain")
close(f)

var content = readFile(path)
print content
print contains_str(content, "again")

system "rm /tmp/jaithon_io_test.txt"
