CC = g++
LIBFLAGS = -c
CFLAGS = -Wall -W -fopenmp -O3 -pedantic -std=c++0x
LFLAGS =
LLIBS =

NVCC = nvcc
NVLIBFLAGS = -c -std=c++11 -rdc=true -arch=sm_30 --expt-relaxed-constexpr
NVFLAGS = -Wno-deprecated-gpu-targets
NVDEBUG = -g -G

CUDAFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart
OPENCVFLAGS = `pkg-config --libs opencv`

all:
	make interlacer

clean:
	rm -f bin/*

interlacer: main.cu interlacing.h
	$(NVCC) $(NVLIBFLAGS) $(OPENCVFLAGS) $(NVDEBUG) interlacing.cu -o bin/interlacing.o $(NVFLAGS)
	$(NVCC) $(NVLIBFLAGS) $(OPENCVFLAGS) $(NVDEBUG) main.cu -o bin/main.o $(NVFLAGS)
	$(NVCC) $(NVDEBUG) $(OPENCVFLAGS) bin/main.o bin/interlacing.o -o bin/interlacer $(NVFLAGS)
