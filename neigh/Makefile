include ../Makefile.common

OBJ = ../common/libcommon.so

all: neigh_bpa_cpu_timer neigh_bpa_gpu_timer neigh_bpa_posix_timer neigh_tpa_cpu_timer neigh_tpa_gpu_timer neigh_tpa_posix_timer

neigh_bpa_cpu_timer: neigh.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) $^ -o $@

neigh_bpa_gpu_timer: neigh.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -D GPU_TIMER $^ -o $@

neigh_bpa_posix_timer: neigh.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -D POSIX_TIMER -lrt $^ -o $@

neigh_tpa_cpu_timer: neigh.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -D COMPUTE_TPA $^ -o $@

neigh_tpa_gpu_timer: neigh.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -D COMPUTE_TPA -D GPU_TIMER $^ -o $@

neigh_tpa_posix_timer: neigh.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -D COMPUTE_TPA -D POSIX_TIMER -lrt $^ -o $@

clean:
	rm -f neigh_bpa_* neigh_tpa_* *.log *.data
