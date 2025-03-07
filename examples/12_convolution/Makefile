# ------------------------------

HIPCC    = hipcc
OFFLOAD_ARCH ?= gfx90a
HIPFLAGS      = --offload-arch=$(OFFLOAD_ARCH)

# ------------------------------

convolution: convolution.o
	$(HIPCC) $(HIPFLAGS) convolution.o -o convolution

convolution.o: convolution.cpp
	$(HIPCC) $(HIPFLAGS) -c convolution.cpp

.PHONY: clean

clean:
	rm -f convolution *.o
