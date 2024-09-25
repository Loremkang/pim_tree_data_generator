CC := g++

all: data_generator

data_generator: ./src/data_generator.cpp ./src/zipfian_generator.hpp ./include makefile
	$(CC) -o data_generator ./src/data_generator.cpp -Iinclude -isystem parlaylib/include -isystem argparse/include -std=c++17 -lpthread -O3 -Wall -Wconversion -Wpedantic

.PHONY: clean

clean:
	rm -f data_generator
	