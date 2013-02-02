NAME='matrix'

CCFLAGS=-pipe
DFLAGS=$(CCFLAGS) -ggdb -Wall
RFLAGS=$(CCFLAGS) -O2
default: debug 

debug: clean
		gcc $(DFLAGS) -o $(NAME) matrix.c

release: clean
		gcc $(RFLAGS) -o $(NAME) matrix.c

clean:
		rm -rf *.o
		rm -rf $(NAME)
