ICC=icc
ICC_FLAGS=-std=gnu99 -xSSE2 -vec_report1 -vec-threshold0

LIBS=-lrt -lm

.PHONY: clean run

CODE_FILES=$(wildcard *.c)

default: matrix

matrix: $(CODE_FILES)
	$(ICC) $(ICC_FLAGS) -o $@ $^ $(LIBS)

run: matrix
	./$<

clean:
	-rm -rf matrix
