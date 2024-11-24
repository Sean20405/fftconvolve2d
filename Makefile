CXX := g++
INCLUDE_FLAGS := $(shell python3 -m pybind11 --includes) $(shell python3-config --includes)
LD_FLAGS := $(shell python3-config --ldflags)
CXX_FLAGS := -O3 -Wall -shared -fPIC -std=c++11
OUTPUT := fft$(shell python3-config --extension-suffix)

.PHONY: default
default: $(OUTPUT)
$(OUTPUT): src/fft.cpp src/fft.hpp pocketfft.o
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $(OUTPUT) $(LD_FLAGS)
	rm -f pocketfft.o

.PHONY: pocketfft.o
pocketfft.o: src/pocketfft/pocketfft.c
	gcc -c src/pocketfft/pocketfft.c -o pocketfft.o

.PHONY: test
test: tests/test_*.py 
	python -m pytest tests/

.PHONY: clean
clean:
	rm -rf *.so *.o __pycache__