NAME=matrix

BUILD_DEFINE=BUILD_NAME
CC=gcc

CC_FLAGS=-pipe -std=gnu99 -DUSE_TCMALLOC
CC_DEBUG_FLAGS=-ggdb -Wall -D$(BUILD_DEFINE)="\"GCC debug\""
CC_RELEASE_FLAGS=-O2 -D$(BUILD_DEFINE)="\"GCC\""

ICC=icc
ICC_FLAGS=-std=gnu99
ICC_NO_OPENMP_FLAGS=-vec -axSSE2 -xSSE2 -vec_report3 -vec-threshold0 -D$(BUILD_DEFINE)="\"ICC\""
ICC_OPENMP_FLAGS=-openmp -no-vec -D$(BUILD_DEFINE)="\"ICC OpenMP\""

LIBS=-lrt -ltcmalloc

default: all

all: clean gcc-debug gcc icc openmp

run: all
	@echo 'Running matrix multiplicators'
	@./matrix-gcc
	@./matrix-icc
	@./matrix-openmp
	
gcc-debug:
	@echo Building $@.
	@$(CC) $(CC_FLAGS) $(CC_DEBUG_FLAGS) -o $(NAME)-$@ *.c $(LIBS)

gcc:
	@echo Building $@.
	@$(CC) $(CC_FLAGS) $(CC_RELEASE_FLAGS) -o $(NAME)-$@ *.c $(LIBS)

icc:
	@echo Building $@. Vectorization log is show below.
	@$(ICC) $(ICC_FLAGS) $(ICC_NO_OPENMP_FLAGS) -o $(NAME)-$@ *.c $(LIBS) 2>&1 | grep VECTORIZED

openmp:
	@echo Building $@.
	@$(ICC) $(ICC_FLAGS) $(ICC_OPENMP_FLAGS) -o $(NAME)-$@ *.c $(LIBS)


clean:
	@echo Cleaning...
	@rm -rf $(NAME)-gcc $(NAME)-gcc-debug $(NAME)-icc $(NAME)-openmp
