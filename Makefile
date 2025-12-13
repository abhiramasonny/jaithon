TARGET = jaithon

CC = gcc
CFLAGS = -Wall -Wextra
LIBS = -lreadline -lm -framework Cocoa -framework Metal -framework QuartzCore

SRC_DIR = src

SOURCES = $(SRC_DIR)/cli/main.c \
          $(SRC_DIR)/core/runtime.c \
          $(SRC_DIR)/core/gui.m \
          $(SRC_DIR)/core/parallel.c \
          $(SRC_DIR)/core/gpu.m \
          $(SRC_DIR)/lang/lexer.c \
          $(SRC_DIR)/lang/parser.c \
          $(SRC_DIR)/vm/vm.c \
          $(SRC_DIR)/vm/compiler.c \
          $(SRC_DIR)/vm/bytecode.c

HEADERS = $(SRC_DIR)/core/runtime.h \
          $(SRC_DIR)/core/parallel.h \
          $(SRC_DIR)/lang/lexer.h \
          $(SRC_DIR)/lang/parser.h \
          $(SRC_DIR)/vm/vm.h \
          $(SRC_DIR)/vm/compiler.h \
          $(SRC_DIR)/vm/bytecode.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
