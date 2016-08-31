all: test

test: APSPtest.o MatUtil.o Floyd.o
	mpicc -o $@ $?

%.o: %.c
	mpicc -c $<

clean:
	rm *.o test
