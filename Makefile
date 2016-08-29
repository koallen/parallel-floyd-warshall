all: test

test: APSPtest.o MatUtil.o
	gcc-6 -o $@ $?

%.o: %.c
	gcc-6 -c $<

clean:
	rm *.o test
