all: test

test: APSPtest.o MatUtil.o Floyd.o
	mpicc -o $@ $?

%.o: %.c
	mpicc -O3 -std=c99 -c $<

clean:
	rm *.o test
