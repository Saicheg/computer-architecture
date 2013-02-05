NAME=matrix

CC=gcc
CC_FLAGS=-pipe -std=gnu99
CC_DEBUG_FLAGS=-ggdb -Wall -D$(BUILD_DEFINE)="\"GCC debug\""
CC_RELEASE_FLAGS=-O2 -D$(BUILD_DEFINE)="\"GCC\""

ICC=icc
ICC_FLAGS=-std=gnu99
ICC_NO_OPENMP_FLAGS=-vec -axSSE2 -xSSE2 -vec_report3 -vec-threshold0
ICC_OPENMP_FLAGS=-openmp -no-vec

LIBS=-lrt

BUILD_DEFINE=BUILD_NAME

default: all

all: clean gcc-debug gcc icc openmp

run: all
	./matrix-gcc
	./matrix-icc
	./matrix-openmp
	
gcc-debug:
	$(CC) $(CC_FLAGS) $(CC_DEBUG_FLAGS) -o $(NAME)-$@ *.c $(LIBS)

gcc:
	$(CC) $(CC_FLAGS) $(CC_RELEASE_FLAGS) -o $(NAME)-$@ *.c $(LIBS) -D$(BUILD_DEFINE)="\"GCC\""

icc:
	$(ICC) $(ICC_FLAGS) $(ICC_NO_OPENMP_FLAGS) -o $(NAME)-$@ *.c $(LIBS) -D${BUILD_DEFINE)="\"ICC\""

openmp:
	$(ICC) $(ICC_FLAGS) $(ICC_OPENMP_FLAGS) -o $(NAME)-$@ *.c $(LIBS) -D$(BUILD_DEFINE)="\"ICC OpenMP\""


clean:
	rm -rf $(NAME)-gcc $(NAME)-gcc-debug $(NAME)-icc $(NAME)-openmp
