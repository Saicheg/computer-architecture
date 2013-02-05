NAME=matrix

CC=gcc
CC_FLAGS=-pipe
CC_DEBUG_FLAGS=-ggdb -Wall
CC_RELEASE_FLAGS=-O2

ICC=icc
ICC_FLAGS=-vec -axSSE2 -vec_report3

LIBS=-lrt

default: gcc-debug

gcc-debug:
	$(CC) $(CC_FLAGS) $(CC_DEBUG_FLAGS) -o $(NAME)-$@ *.c $(LIBS)

gcc:
	$(CC) $(CC_FLAGS) $(CC_RELEASE_FLAGS) -o $(NAME)-$@ *.c $(LIBS)

icc:
	$(ICC) $(ICC_FLAGS) -o $(NAME)-$@ *.c $(LIBS)

clean:
	rm -rf $(NAME)-gcc $(NAME)-gcc-debug $(NAME)-icc
