# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx90a

# ------------------------------

convolution: convolution.o
	${HIPCC} ${HIPFLAGS} convolution.o -o convolution

convolution.o: convolution.cpp
	${HIPCC} ${HIPFLAGS} -c convolution.cpp

.PHONY: clean

clean:
	rm -f convolution *.o
