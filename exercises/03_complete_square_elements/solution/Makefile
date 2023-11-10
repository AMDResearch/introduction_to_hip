# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx908

# ------------------------------

square_elements: square_elements.o
	${HIPCC} square_elements.o -o square_elements

square_elements.o: square_elements.cpp
	${HIPCC} -c square_elements.cpp

.PHONY: clean

clean:
	rm -f square_elements *.o
