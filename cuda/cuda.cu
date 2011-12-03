/*
 * Contact decomposition of the hertz pairwise kernel. 
 * 
 * We take two datasets (from file) inputs (per-particle and contact data) and
 * expected output following a pairwise calculation.
 * 
 * We prepare an array of struct read-dataset and a struct of array
 * write-dataset. The kernel itself acts on individual contacts and writes out
 * *delta* [force] and [torque] values that must be postprocessed to give the
 * expected output.
 *
 * We sum through this indirection by first creating an inverse mapping and
 * running a separate collect kernel over the [force] and [torque] arrays.
 *
 */

//#define MAP_BUILD_CHECK //< bounds and sanity checking in build_inverse_map
//#define KERNEL_PRINT    //< debug printing in kernel
//#define DEBUG           //< add (i,j) index information to struct

#include "check_result_vector.h"
#include "cuda_common.h"
#include "framework.h"
#include "hertz_constants.h"
#include "pair_interaction.h"
#include "inverse_map.h"
#include <sstream>

using namespace std;

// --------------------------------------------------------------------------
// DEVICE KERNEL
// --------------------------------------------------------------------------

// AoS read-dataset for device
// TODO: rearrange struct
// TODO(1): take out x, v, and omega
// TODO(2): move radius, mass and type to constant
struct contact {
#ifdef DEBUG
  int i; int j;
#endif
  double xi[3];     double xj[3];
  double vi[3];     double vj[3];
  double omegai[3]; double omegaj[3];
  double radiusi;   double radiusj;
  double massi;     double massj;
  int    typei;     int typej;
};

__global__ void aos_kernel(
    int ncontacts,
    struct contact *aos,
    double *force,
    double *torque,
    double *torquej,
    double  *shear) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < ncontacts) {
    struct contact c = aos[idx];

    pair_interaction(
#ifdef DEBUG
        c.i,c.j,
#endif
        c.xi, c.xj,
        c.vi, c.vj,
        c.omegai, c.omegaj,
        c.radiusi, c.radiusj,
        c.massi, c.massj,
        c.typei, c.typej,
        &shear[(idx*3)],
        &force[(idx*3)], /*forcej is */NULL,
        &torque[(idx*3)], &torquej[(idx*3)]);
  }
}

__global__ void collect_kernel(
  int nparticles,
  double *force_delta, double *torquei_delta, double *torquej_delta,
  int *ioffset, int *icount, int *imapinv,
  int *joffset, int *jcount, int *jmapinv,
  //outputs
  double *force, double *torque) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < nparticles) {
    double fdelta[3] = {0.0, 0.0, 0.0};
    double tdelta[3] = {0.0, 0.0, 0.0};

    int ioff = ioffset[idx];
    for (int i=0; i<icount[idx]; i++) {
      int e = imapinv[ioff+i];
      fdelta[0] += force_delta[(e*3)  ];
      fdelta[1] += force_delta[(e*3)+1];
      fdelta[2] += force_delta[(e*3)+2];

      tdelta[0] += torquei_delta[(e*3)  ];
      tdelta[1] += torquei_delta[(e*3)+1];
      tdelta[2] += torquei_delta[(e*3)+2];
    }

    int joff = joffset[idx];
    for (int i=0; i<jcount[idx]; i++) {
      int e = jmapinv[joff+i];

      fdelta[0] -= force_delta[(e*3)  ];
      fdelta[1] -= force_delta[(e*3)+1];
      fdelta[2] -= force_delta[(e*3)+2];

      tdelta[0] += torquej_delta[(e*3)  ];
      tdelta[1] += torquej_delta[(e*3)+1];
      tdelta[2] += torquej_delta[(e*3)+2];
    }

    //output
    force[(idx*3)]   += fdelta[0];
    force[(idx*3)+1] += fdelta[1];
    force[(idx*3)+2] += fdelta[2];

    torque[(idx*3)]   += tdelta[0];
    torque[(idx*3)+1] += tdelta[1];
    torque[(idx*3)+2] += tdelta[2];
  }
}

// --------------------------------------------------------------------------
// RUN
// --------------------------------------------------------------------------

void run(struct params *input, int num_iter) {

  //--------------------
  // One-time only costs
  //--------------------

  //Hertz constants
  one_time.push_back(SimpleTimer("hertz_consts"));
  one_time.back().start();
  setup_hertz_constants(input);
  one_time.back().stop_and_add_to_total();

  //AoS generate
  one_time.push_back(SimpleTimer("aos_gen"));
  one_time.back().start();
  struct contact *aos = new contact[input->nedge];
  for (int e=0; e<input->nedge; e++) {
    int i = input->edge[(e*2)];
    int j = input->edge[(e*2)+1];

#ifdef DEBUG
    aos[e].i = i;
    aos[e].j = j;
#endif

    aos[e].xi[0] = input->x[(i*3)];
    aos[e].xi[1] = input->x[(i*3)+1];
    aos[e].xi[2] = input->x[(i*3)+2];
    aos[e].xj[0] = input->x[(j*3)];
    aos[e].xj[1] = input->x[(j*3)+1];
    aos[e].xj[2] = input->x[(j*3)+2];

    aos[e].vi[0] = input->v[(i*3)];
    aos[e].vi[1] = input->v[(i*3)+1];
    aos[e].vi[2] = input->v[(i*3)+2];
    aos[e].vj[0] = input->v[(j*3)];
    aos[e].vj[1] = input->v[(j*3)+1];
    aos[e].vj[2] = input->v[(j*3)+2];

    aos[e].omegai[0] = input->omega[(i*3)];
    aos[e].omegai[1] = input->omega[(i*3)+1];
    aos[e].omegai[2] = input->omega[(i*3)+2];
    aos[e].omegaj[0] = input->omega[(j*3)];
    aos[e].omegaj[1] = input->omega[(j*3)+1];
    aos[e].omegaj[2] = input->omega[(j*3)+2];

    aos[e].radiusi = input->radius[i];
    aos[e].radiusj = input->radius[j];

    aos[e].massi = input->mass[i];
    aos[e].massj = input->mass[j];

    aos[e].typei = input->type[i];
    aos[e].typej = input->type[j];
  }

  struct contact *d_aos;
  const int d_aos_size = input->nedge * sizeof(struct contact);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_aos, d_aos_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_aos, aos, d_aos_size, cudaMemcpyHostToDevice));
  one_time.back().stop_and_add_to_total();

  one_time.push_back(SimpleTimer("malloc_on_dev"));
  one_time.back().start();
  double *d_force_delta;
  double *d_torquei_delta;
  double *d_torquej_delta;
  const int d_delta_size = input->nedge * 3 *sizeof(double);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_force_delta, d_delta_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_torquei_delta, d_delta_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_torquej_delta, d_delta_size));

  double *d_force;
  double *d_torque;
  const int d_output_size = input->nnode * 3 * sizeof(double);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_force, d_output_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_torque, d_output_size));

  double *d_shear;
  const int d_shear_size = input->nedge * 3 * sizeof(double);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_shear, d_shear_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_shear, input->shear, d_shear_size, cudaMemcpyHostToDevice));
  one_time.back().stop_and_add_to_total();

  one_time.push_back(SimpleTimer("inverse_map_build"));
  one_time.back().start();
  //inverse mappings for (i,j) particle pairs
  int *imap = new int[input->nedge];
  int *jmap = new int[input->nedge];
  for (int e=0; e<input->nedge; e++) {
    imap[e] = input->edge[(e*2)  ];
    jmap[e] = input->edge[(e*2)+1];
  }
  int *ioffset = NULL;
  int *icount = NULL;
  int *imapinv = NULL;
  int *joffset = NULL;
  int *jcount = NULL;
  int *jmapinv = NULL;
  build_inverse_map(imap, input->nedge, input->nnode,
    ioffset, icount, imapinv);
  build_inverse_map(jmap, input->nedge, input->nnode,
    joffset, jcount, jmapinv);

  int *d_ioffset;
  int *d_icount;
  int *d_imapinv;
  int *d_joffset;
  int *d_jcount;
  int *d_jmapinv;
  const int d_nnode_size = input->nnode * sizeof(int);
  const int d_nedge_size = input->nedge * sizeof(int);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_ioffset, d_nnode_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_icount, d_nnode_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_imapinv, d_nedge_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_joffset, d_nnode_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_jcount, d_nnode_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_jmapinv, d_nedge_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_ioffset, ioffset, d_nnode_size, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_icount, icount, d_nnode_size, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_imapinv, imapinv, d_nedge_size, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_joffset, joffset, d_nnode_size, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_jcount, jcount, d_nnode_size, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_jmapinv, jmapinv, d_nedge_size, cudaMemcpyHostToDevice));
  one_time.back().stop_and_add_to_total();

  //TODO(1): copy real x, v, omega in PREPROCESS
  //These are dummy structures just for timing
  double *d_fake_x;
  double *d_fake_v;
  double *d_fake_omega;
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_fake_x, d_output_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_fake_v, d_output_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_fake_omega, d_output_size));

  //--------------------
  // Per-iteration costs
  //--------------------

  per_iter.push_back(SimpleTimer("aos_memcpy_to_dev"));
  per_iter.push_back(SimpleTimer("compute_kernel"));
  per_iter.push_back(SimpleTimer("collect_kernel"));
  per_iter.push_back(SimpleTimer("result_fetch"));

  for (int run=0; run<num_iter; run++) {
    //PREPROCESSING
    //copy across structures that change between kernel invocations, 
    //reset delta structures (force/torque).
    per_iter[0].start();
    //TODO(1): just copy dummy structures for timing
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_fake_x, input->x, d_output_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_fake_v, input->v, d_output_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_fake_omega, input->omega, d_output_size, cudaMemcpyHostToDevice));

    ASSERT_NO_CUDA_ERROR(
      cudaMemset((void *)d_force_delta, 0, d_delta_size));
    ASSERT_NO_CUDA_ERROR(
      cudaMemset((void *)d_torquei_delta, 0, d_delta_size));
    ASSERT_NO_CUDA_ERROR(
      cudaMemset((void *)d_torquej_delta, 0, d_delta_size));
    per_iter[0].stop_and_add_to_total();

    //NB: safe to omit from preprocess costs because they do not change between
    //kernel invocations.
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_force, input->force, d_output_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_torque, input->torque, d_output_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_shear, input->shear, d_shear_size, cudaMemcpyHostToDevice));

    //-----------------------------------------------------------------------

    //KERNEL INVOCATION
    //pairwise kernel computes delta values (force/torque) and shear
    //gather kernel produces force/torque results
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Pre-kernel error: %s.\n", cudaGetErrorString(err));
      exit(1);
    }

#ifdef KERNEL_PRINT
    cudaPrintfInit();
#endif

    const int blockSize = 128;
    dim3 gridSize((input->nedge / blockSize)+1);
    per_iter[1].start();
    aos_kernel<<<gridSize, blockSize>>>(
      input->nedge,
      d_aos,
      d_force_delta, d_torquei_delta, d_torquej_delta, d_shear);
    cudaThreadSynchronize();
    per_iter[1].stop_and_add_to_total();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Post-kernel error: %s.\n", cudaGetErrorString(err));
      exit(1);
    }

    const int gatherBlockSize = 128;
    dim3 gatherGridSize((input->nnode / gatherBlockSize)+1);
    per_iter[2].start();
    collect_kernel<<<gatherGridSize, gatherBlockSize>>>(
      input->nnode,
      d_force_delta, d_torquei_delta, d_torquej_delta,
      d_ioffset, d_icount, d_imapinv,
      d_joffset, d_jcount, d_jmapinv,
      d_force, d_torque);
    cudaThreadSynchronize();
    per_iter[2].stop_and_add_to_total();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Post-gather error: %s.\n", cudaGetErrorString(err));
      exit(1);
    }

#ifdef KERNEL_PRINT
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
#endif

    //-----------------------------------------------------------------------

    //POSTPROCESSING
    //memcpy data back to host
    per_iter[3].start();
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(input->force, d_force, d_output_size, cudaMemcpyDeviceToHost));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(input->torque, d_torque, d_output_size, cudaMemcpyDeviceToHost));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(input->shear, d_shear, d_shear_size, cudaMemcpyDeviceToHost));
    per_iter[3].stop_and_add_to_total();

    //-----------------------------------------------------------------------

    //CHECKING
    //only check results the first time around
    if (run == 0) {
      for (int n=0; n<input->nnode; n++) {
        std::stringstream out;
        out << "force[" << n << "]";
        check_result_vector(
            out.str().c_str(),
            &input->expected_force[(n*3)], &input->force[(n*3)]);
        out.str("");

        out << "torque[" << n << "]";
        check_result_vector(
            out.str().c_str(),
            &input->expected_torque[(n*3)], &input->torque[(n*3)]);
      }
      for (int n=0; n<input->nedge; n++) {
        stringstream out;
        out << "shear[" << n << "]";
        check_result_vector(
            out.str().c_str(), 
            &input->expected_shear[(n*3)], &input->shear[(n*3)]);
      }
    }
  }

  //CLEANUP
  cudaFree(d_aos);
  cudaFree(d_force_delta);
  cudaFree(d_torquei_delta);
  cudaFree(d_torquej_delta);
  cudaFree(d_shear);
  cudaFree(d_force);
  cudaFree(d_torque);

  cudaFree(d_ioffset);
  cudaFree(d_icount);
  cudaFree(d_imapinv);
  cudaFree(d_joffset);
  cudaFree(d_jcount);
  cudaFree(d_jmapinv);

  //TODO(1): free dummy structures
  cudaFree(d_fake_x);
  cudaFree(d_fake_v);
  cudaFree(d_fake_omega);
}
