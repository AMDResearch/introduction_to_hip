# ------------------------------

HIPCC    = hipcc
OFFLOAD_ARCH ?= gfx90a
HIPFLAGS      = --offload-arch=$(OFFLOAD_ARCH)

# ------------------------------

vector_addition: vector_addition.o
	$(HIPCC) $(HIPFLAGS) vector_addition.o -o vector_addition

vector_addition.o: vector_addition.cu
	$(HIPCC) $(HIPFLAGS) -c vector_addition.cu

.PHONY: clean

clean:
	rm -f vector_addition *.o
