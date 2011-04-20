#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#ifdef KERNEL_PRINT
  #warning "Enabling cuPrintf in cuda kernels. Timing may not be accurate."
  #include "cuPrintf.cu"
#endif

#define ASSERT_NO_CUDA_ERROR( callReturningErrorstatus ) {     \
  cudaError_t err = callReturningErrorstatus;                  \
  if (err != cudaSuccess) {                                    \
    fprintf(stderr,                                            \
            "Cuda error (%s/%d) in file '%s' in line %i\n",    \
            cudaGetErrorString(err), err, __FILE__, __LINE__); \
    exit(1);                                                   \
  }                                                            \
} while(0);

#endif
