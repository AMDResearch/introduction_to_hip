# ------------------------------

HIPCC         = hipcc
OFFLOAD_ARCH ?= gfx90a
HIPFLAGS      = --offload-arch=$(OFFLOAD_ARCH)
HIPLIBS       = -L$(ROCM_PATH)/hipblas/lib -lhipblas

# ------------------------------

matrix_multiply: matrix_multiply.o
	$(HIPCC) $(HIPFLAGS) $(HIPLIBS) matrix_multiply.o -o matrix_multiply

matrix_multiply.o: matrix_multiply.cpp
	$(HIPCC) $(HIPFLAGS) -c matrix_multiply.cpp

.PHONY: clean

clean:
	rm -f matrix_multiply *.o
