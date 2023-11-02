# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx908

# ------------------------------

vector_addition: vector_addition.o
	${HIPCC} vector_addition.o -o vector_addition

vector_addition.o: vector_addition.cu
	${HIPCC} -c vector_addition.cu

.PHONY: clean

clean:
	rm -f vector_addition *.o
