NAME='matrix'

CCFLAGS=-O2 -pipe -Wall
DFLAGS=$(CFLAGS) -ggdb

default: release

debug: clean
		gcc $(DFLAGS) -o $(NAME) matrix.c

release: clean
		gcc $(CCFLAGS) -o $(NAME) matrix.c

clean:
		rm -rf *.o
		rm -rf $(NAME)
