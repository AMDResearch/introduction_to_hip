# ------------------------------

HIPCC         = hipcc
OFFLOAD_ARCH ?= gfx90a
HIPFLAGS      = --offload-arch=$(OFFLOAD_ARCH)

# ------------------------------

stencil: stencil.o
	$(HIPCC) $(HIPFLAGS) stencil.o -o stencil

stencil.o: stencil.cpp
	$(HIPCC) $(HIPFLAGS) -c stencil.cpp

.PHONY: clean

clean:
	rm -f stencil *.o
