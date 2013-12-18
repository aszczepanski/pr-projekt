
all: main.o
	g++ -o main main.o -fopenmp

main.o: main.cpp
	g++ -o main.o -c main.cpp -Wall -fopenmp

.PHONY: clean
clean:
	-rm -rf main *.o
