NAME=matrix

BUILD_DEFINE=BUILD_NAME

CC=gcc
CC_FLAGS=-pipe -std=gnu99 -DUSE_TCMALLOC
CC_DEBUG_FLAGS=-ggdb -Wall -D$(BUILD_DEFINE)="\"GCC debug\""
CC_RELEASE_FLAGS=-O2 -D$(BUILD_DEFINE)="\"GCC\""
CC_OPENMP_FLAGS=-O2 -fopenmp -D$(BUILD_DEFINE)="\"GCC OpenMP\""

ICC=icc
ICC_FLAGS=-std=gnu99 -vec -xSSE2 -vec_report3 -vec-threshold0
ICC_RELEASE_FLAGS=-D$(BUILD_DEFINE)="\"ICC\""
ICC_OPENMP_FLAGS=-openmp -openmp-report=2 -D$(BUILD_DEFINE)="\"ICC OpenMP\""

LIBS=-lrt -ltcmalloc

default: all

all: clean gcc-debug gcc icc gcc-openmp icc-openmp

run: all
	@echo 'Running.'
	@./matrix-gcc
	@./matrix-icc
	@./matrix-gcc-openmp
	@./matrix-icc-openmp
	
gcc-debug:
	@echo Building $@.
	@$(CC) $(CC_FLAGS) $(CC_DEBUG_FLAGS) -o $(NAME)-$@ *.c $(LIBS)

gcc:
	@echo Building $@.
	@$(CC) $(CC_FLAGS) $(CC_RELEASE_FLAGS) -o $(NAME)-$@ *.c $(LIBS)

gcc-openmp:
	@echo Building $@.
	@$(CC) $(CC_FLAGS) $(CC_OPENMP_FLAGS) -o $(NAME)-$@ *.c $(LIBS)

icc:
	@echo Building $@. Vectorization log is show below.
	@$(ICC) $(ICC_FLAGS) $(ICC_RELEASE_FLAGS) -o $(NAME)-$@ *.c $(LIBS) 2>&1 | grep 'VECTORIZED'

icc-openmp:
	@echo Building $@.
	@$(ICC) $(ICC_FLAGS) $(ICC_OPENMP_FLAGS) -o $(NAME)-$@ *.c $(LIBS) 2>&1 | grep -E '(VECTORIZED|PARALLELIZED)'

clean:
	@echo Cleaning...
	@rm -rf $(NAME)-gcc $(NAME)-gcc-debug $(NAME)-gcc-openmp $(NAME)-icc $(NAME)-icc-openmp
