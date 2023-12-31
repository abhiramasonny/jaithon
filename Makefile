TARGET = jaithon

CC = gcc
CFLAGS = -Wall
LIBS = -lreadline

SOURCES = src/interpreter.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LIBS) -lm

# Clean the build
clean:
	rm -f $(TARGET)
