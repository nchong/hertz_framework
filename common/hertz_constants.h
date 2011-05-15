#ifndef HERTZ_CONSTANTS_H
#define HERTZ_CONSTANTS_H

#define sqrtFiveOverSix 0.91287092917527685576161630466800355658790782499663875

#ifdef __CUDACC__
#include "cuda_common.h"

__constant__ double d_dt;
__constant__ double d_nktv2p;
__constant__ double d_yeff;
__constant__ double d_geff;
__constant__ double d_betaeff;
__constant__ double d_coeffFrict;

void setup_hertz_constants() {
  double dt = 0.00001;
  double nktv2p = 1;
  double yeff = 3134796.2382445144467056;
  double geff = 556173.5261401557363570;
  double betaeff = -0.3578571305033167;
  double coeffFrict = 0.5;
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpyToSymbol("d_dt", &dt, sizeof(double),
      0, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpyToSymbol("d_nktv2p", &nktv2p, sizeof(double),
      0, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpyToSymbol("d_yeff", &yeff, sizeof(double),
      0, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpyToSymbol("d_geff", &geff, sizeof(double),
      0, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpyToSymbol("d_betaeff", &betaeff, sizeof(double),
      0, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpyToSymbol("d_coeffFrict", &coeffFrict, sizeof(double),
      0, cudaMemcpyHostToDevice));
}
#endif

#endif
