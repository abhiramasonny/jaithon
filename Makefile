TARGET = jaithon

CC = gcc
CFLAGS = -Wall -Wextra
LIBS = -lreadline -lm -framework Cocoa

SRC_DIR = src

SOURCES = $(SRC_DIR)/cli/main.c \
          $(SRC_DIR)/core/runtime.c \
          $(SRC_DIR)/core/gui.m \
          $(SRC_DIR)/lang/lexer.c \
          $(SRC_DIR)/lang/parser.c

HEADERS = $(SRC_DIR)/core/runtime.h \
          $(SRC_DIR)/lang/lexer.h \
          $(SRC_DIR)/lang/parser.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
