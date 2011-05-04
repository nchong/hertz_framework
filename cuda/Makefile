include ../Makefile.common

# NVCCFLAGS+=-g

OBJ = inverse_map.o ../common/libcommon.so

all: cuda_cpu_timer cuda_gpu_timer cuda_posix_timer

cuda_cpu_timer: cuda.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) $^ -o $@

cuda_gpu_timer: cuda.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -D GPU_TIMER $^ -o $@

cuda_posix_timer: cuda.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -D POSIX_TIMER -lrt $^ -o $@

clean:
	rm -f inverse_map.o cuda_cpu_timer cuda_gpu_timer cuda_posix_timer *.log *.data
