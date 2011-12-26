include ../Makefile.common

# NVCCFLAGS+=-g

OBJ = ../common/libcommon.so

all: cuda_posix_timer

cuda_cpu_timer: cuda.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) $^ -o $@

cuda_gpu_timer: cuda.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -D GPU_TIMER $^ -o $@

cuda_posix_timer: cuda.cu
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -D POSIX_TIMER -L../common -lrt -lcommon $^ -o $@

clean:
	rm -f cuda_cpu_timer cuda_gpu_timer cuda_posix_timer *.log *.data
