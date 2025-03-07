# ------------------------------

HIPCC    = hipcc
OFFLOAD_ARCH ?= gfx90a
HIPFLAGS      = --offload-arch=$(OFFLOAD_ARCH)

# ------------------------------

square_elements: square_elements.o
	$(HIPCC) $(HIPFLAGS) square_elements.o -o square_elements

square_elements.o: square_elements.cpp
	$(HIPCC) $(HIPFLAGS) -c square_elements.cpp

.PHONY: clean

clean:
	rm -f square_elements *.o
