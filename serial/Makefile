include ../Makefile.common

OBJ = ../common/libcommon.so

all: serial_cpu_timer serial_posix_timer

serial_cpu_timer: serial.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) -I $(INCLUDEDIR) $^ -o $@

#serial_gpu_timer: serial.cu $(OBJ)
#	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -D GPU_TIMER  $^ -o $@

serial_posix_timer: serial.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) -I $(INCLUDEDIR) -D POSIX_TIMER -lrt $^ -o $@

clean:
	rm -f serial_cpu_timer serial_gpu_timer serial_posix_timer *.log *.data
