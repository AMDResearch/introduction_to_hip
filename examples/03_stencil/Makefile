# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx908

# ------------------------------

stencil: stencil.o
	${HIPCC} stencil.o -o stencil

stencil.o: stencil.cpp
	${HIPCC} -c stencil.cpp

.PHONY: clean

clean:
	rm -f stencil *.o
