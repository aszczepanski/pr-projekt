
OPT_FLAGS = -O2

main: main.cpp
	g++ -o main main.cpp -Wall -fopenmp -std=c++11 $(OPT_FLAGS)

.PHONY: clean
clean:
	-rm -rf main *.o
