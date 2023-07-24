TARGET = interpreter

CC = gcc

SOURCES = interpreter.c

all: $(TARGET)
$(TARGET): $(SOURCES)
	$(CC) -o $(TARGET) $(SOURCES)

# Clean the build
clean:
	rm -f $(TARGET)
