
CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic

all:
	c++ $(CXXFLAGS) expt.cpp -o expt
	mkdir -p output
	./expt

clean:
	rm -rf expt

.PHONY: all expt
