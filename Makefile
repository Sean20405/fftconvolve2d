CXX := g++
INCLUDE_FLAGS := $(shell python3 -m pybind11 --includes) $(shell python3-config --includes)
LD_FLAGS := $(shell python3-config --ldflags)
CXX_FLAGS := -Ofast -Wall -shared -fPIC -std=c++11
OUTPUT := fft$(shell python3-config --extension-suffix)

.PHONY: default
default: $(OUTPUT)
$(OUTPUT): src/fft.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $(OUTPUT) $(LD_FLAGS)

.PHONY: test
test: tests/test_*.py $(OUTPUT)
	python -m pytest tests/

.PHONY: t_cpp
t_cpp: src/main.cpp src/fft.cpp
	$(CXX) -Ofast -Wall -std=c++11 $^ -o main
	./main

.PHONY: clean
clean:
	rm -rf *.so *.o __pycache__