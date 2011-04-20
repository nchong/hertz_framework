include ../Makefile.common

OBJ = ../common/libcommon.so

all: serial_cpu_timer serial_gpu_timer

serial_cpu_timer: serial.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) -I $(INCLUDEDIR) $^ -o $@

serial_gpu_timer: serial.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -D GPU_TIMER -I $(INCLUDEDIR) $^ -o $@

clean:
	rm -f serial_cpu_timer serial_gpu_timer
