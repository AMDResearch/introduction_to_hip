# ------------------------------

HIPCC    = hipcc
OFFLOAD_ARCH ?= gfx90a
HIPFLAGS      = --offload-arch=$(OFFLOAD_ARCH)

# ------------------------------

pingpong: pingpong.o
	$(HIPCC) $(HIPFLAGS) pingpong.o -o pingpong

pingpong.o: pingpong.cpp
	$(HIPCC) $(HIPFLAGS) -c pingpong.cpp

.PHONY: clean

clean:
	rm -f pingpong *.o
