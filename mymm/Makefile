CC = gc
CXX = g++
NVCC = nvcc
CFLAGS = -Wall -O3 -std=c++11 -DCUDA
NVCCFLAGS= -std=c++11 -c -arch=sm_30
# cuda library
CULIBS= -lopenblas -lpthread -lcudart -lcublas
# general library
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
# path of general library
LIBDIRS=-L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu/
# path of general include headers
INCDIRS=-I/usr/local/cuda/include -I/usr/include/opencv2

BIN = main 
OBJ = 
CUOBJ = cu.o
CUBIN = 

all: $(CUOBJ) $(BIN) $(OBJ) $(CUBIN) 
cu.o: mm.cu
main: main.cpp 

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^) cu.o  $(LIBDIRS) $(INCDIRS) $(CULIBS) $(LIBS) 

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~

