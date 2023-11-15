# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx908

# ------------------------------

pingpong: pingpong.o
	${HIPCC} pingpong.o -o pingpong

pingpong.o: pingpong.cpp
	${HIPCC} -c pingpong.cpp

.PHONY: clean

clean:
	rm -f pingpong *.o
