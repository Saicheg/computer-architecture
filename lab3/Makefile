CC=gcc
CFLAGS_COMMON=--std=gnu99
CFLAGS=$(CFLAGS_COMMON) -Wl,--as-needed -ggdb -O2
LIBS=-lm

SOURCE_FILES=$(wildcard *.c)
DEPENDENCIES=dependencies.d

.PHONY: clean prepare build run


default: build
	
build: prepare matrix

matrix: matrix.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
	
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

prepare: $(SOURCE_FILES)
	gcc $(CFLAGS_COMMON) -MM $(CFLAGS) $^ > $(DEPENDENCIES)

run: build
	./matrix

clean:
	-rm -rf matrix *.o $(DEPENDENCIES)

-include $(DEPENDENCIES)
