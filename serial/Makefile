include ../Makefile.common

OBJ = ../common/libcommon.so

all: serial_posix_timer serial_omp

serial_posix_timer: serial.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) -I $(INCLUDEDIR) -w -ggdb $^ -o $@

serial_omp: serial.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) -openmp -openmp-report -I $(INCLUDEDIR) -D POSIX_TIMER -lrt $^ -o $@

clean:
	rm -f serial_posix_timer serial_omp *.log *.data
