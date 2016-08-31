all: test

test: APSPtest.o MatUtil.o
	mpicc -o $@ $?

%.o: %.c
	mpicc -c $<

clean:
	rm *.o test
