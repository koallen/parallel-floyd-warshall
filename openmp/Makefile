all: test

test: APSPtest.o MatUtil.o Floyd_openMP.o
	gcc -fopenmp -o $@ $?

%.o: %.c
	gcc -std=c99 -c $<

clean:
	rm *.o test
