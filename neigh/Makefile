include ../Makefile.common

# NVCCFLAGS+=-g

OBJ = ../common/libcommon.so

all: neigh_cpu_timer neigh_gpu_timer neigh_posix_timer

neigh_cpu_timer: neigh.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) $^ -o $@

neigh_gpu_timer: neigh.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -D GPU_TIMER $^ -o $@

neigh_posix_timer: neigh.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -D POSIX_TIMER -lrt $^ -o $@

clean:
	rm -f neigh_cpu_timer neigh_gpu_timer neigh_posix_timer *.log *.data
