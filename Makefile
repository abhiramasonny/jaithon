TARGET = interpreter

CC = gcc

SOURCES = src/interpreter.c

all: $(TARGET)
$(TARGET): $(SOURCES)
	$(CC) -o $(TARGET) $(SOURCES)

# Clean the build
clean:
	rm -f $(TARGET)
