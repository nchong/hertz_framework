#include "hertz_constants.h"
#include "hertz_cudaneighlist.h"
#include "pair_interaction.h"
#include "framework.h"

#ifdef TRACE
#warning TRACE enabled: timing will not be accurate
#include "cuPrintf.cu"
#endif

#ifndef MAX_GRID_DIM
#error You need to #define MAX_GRID_DIM (see Makefile.config)
#endif

__device__ int get_gid() {
  return threadIdx.x + (blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.x * gridDim.x);
}

// --------------------------------------------------------------------------
// UNPACK PER-PARTICLE DATA
// --------------------------------------------------------------------------
__global__ void unpack_ro_data(
  int K,
                  int    *valid,
                  int    *dati,    int    *datj,
  double *radius, double *radiusi, double *radiusj,
  double *mass,   double *massi,   double *massj,
  int    *type,   int    *typei,   int    *typej
) {
  int gid = get_gid();
  if (gid < K && valid[gid]) {
    int i = dati[gid]; int j = datj[gid];
    radiusi[gid] = radius[i]; radiusj[gid] = radius[j];
    massi[gid]   = mass[i];   massj[gid]   = mass[j];
    typei[gid]   = type[i];   typej[gid]   = type[j];
  }
}

__global__ void unpack_reload_data(
  int K,
                  int    *valid,
                  int    *dati,    int    *datj,
  double *x,      double *xi,      double *xj,
  double *v,      double *vi,      double *vj,
  double *omega,  double *omegai,  double *omegaj
) {
  int gid = get_gid();
  if (gid < K && valid[gid]) {
    int i = dati[gid]; int j = datj[gid];
    xi[(gid*3)+0] = x[(i*3)+0]; xj[(gid*3)+0] = x[(j*3)+0];
    xi[(gid*3)+1] = x[(i*3)+1]; xj[(gid*3)+1] = x[(j*3)+1];
    xi[(gid*3)+2] = x[(i*3)+2]; xj[(gid*3)+2] = x[(j*3)+2];
    vi[(gid*3)+0] = v[(i*3)+0]; vj[(gid*3)+0] = v[(j*3)+0];
    vi[(gid*3)+1] = v[(i*3)+1]; vj[(gid*3)+1] = v[(j*3)+1];
    vi[(gid*3)+2] = v[(i*3)+2]; vj[(gid*3)+2] = v[(j*3)+2];
    omegai[(gid*3)+0] = omega[(i*3)+0]; omegaj[(gid*3)+0] = omega[(j*3)+0];
    omegai[(gid*3)+1] = omega[(i*3)+1]; omegaj[(gid*3)+1] = omega[(j*3)+1];
    omegai[(gid*3)+2] = omega[(i*3)+2]; omegaj[(gid*3)+2] = omega[(j*3)+2];
  }
}

__global__ void compute(
  //inputs
  int K,
  int    *valid,
#ifdef TRACE
  int    *dati,    int    *datj,
#endif
  double *xi,      double *xj,
  double *vi,      double *vj,
  double *omegai,  double *omegaj,
  double *radiusi, double *radiusj,
  double *massi,   double *massj,
  int    *typei,   int    *typej,
  //inouts
  double *fdelta,
  double *tdeltai, double *tdeltaj,
  double *shear
) {
  int gid = get_gid();
  if (gid < K && valid[gid]) {
    pair_interaction(
#ifdef TRACE
      dati[gid], datj[gid],
#endif
      &xi[gid*3],     &xj[gid*3],
      &vi[gid*3],     &vj[gid*3],
      &omegai[gid*3], &omegaj[gid*3],
      radiusi[gid],   radiusj[gid],
      massi[gid],     massj[gid],
      typei[gid],     typej[gid],
      &shear[gid*3],
      &fdelta[gid*3], /*fdeltaj is*/NULL,
      &tdeltai[gid*3], &tdeltaj[gid*3]
    );
  }
}

__global__ void collect(
  //inputs
  int N,
  double *fdelta,
  double *tdeltai, double *tdeltaj,
            int *off, int *len,
#if HALFNL
  int *tad, int *ffo, int *nel,
#endif
  //inouts
  double *force,
  double *torque
) {
  int gid = get_gid();

  double fsum[3] = {0,0,0};
  double tsum[3] = {0,0,0};
  if (gid < N) {
    int offset = off[gid];
    for (int k=0; k<len[gid]; k++) {
      int idx = offset+k;
      fsum[0] += fdelta[(idx*3)+0];
      fsum[1] += fdelta[(idx*3)+1];
      fsum[2] += fdelta[(idx*3)+2];
      tsum[0] += tdeltai[(idx*3)+0];
      tsum[1] += tdeltai[(idx*3)+1];
      tsum[2] += tdeltai[(idx*3)+2];
    }
#if HALFNL
    offset = ffo[gid];
    for (int k=0; k<nel[gid]; k++) {
      int idx = tad[offset+k];
      fsum[0] -= fdelta[(idx*3)+0];
      fsum[1] -= fdelta[(idx*3)+1];
      fsum[2] -= fdelta[(idx*3)+2];
      tsum[0] += tdeltaj[(idx*3)+0];
      tsum[1] += tdeltaj[(idx*3)+1];
      tsum[2] += tdeltaj[(idx*3)+2];
    }
    force[(gid*3)]    += fsum[0];
    force[(gid*3)+1]  += fsum[1];
    force[(gid*3)+2]  += fsum[2];
    torque[(gid*3)]   += tsum[0];
    torque[(gid*3)+1] += tsum[1];
    torque[(gid*3)+2] += tsum[2];
#endif
  }
}

using namespace std;

// DEVICE STRUCTURES
// INPUTS
// packed         // unpacked(i)     // unpacked(j)
double *d_x;      double *d_xi;      double *d_xj;        // ] reload
double *d_v;      double *d_vi;      double *d_vj;        // ]
double *d_omega;  double *d_omegai;  double *d_omegaj;    // ]
double *d_radius; double *d_radiusi; double *d_radiusj;   // ] ro
double *d_mass;   double *d_massi;   double *d_massj;     // ]
int    *d_type;   int    *d_typei;   int *d_typej;        // ]
// OUTPUTS
// packed         // unpacked(i)     // unpacked(j)
double *d_force;  double *d_fdelta;
double *d_torque; double *d_tdeltai; double *d_tdeltaj;
//                        d_shear in d_nl

void no_cuda_error(const char *errmsg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("[ERROR] %s\n", errmsg);
    printf("[ERROR] %d: %s\n", err, cudaGetErrorString(err));
    size_t free; size_t total;
    if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
      printf("[ERROR] mem free=%zubytes total=%zubytes\n", free, total);
    }
    exit(1);
  }
}

#define NLEN(type,arity) (nparticles*arity*sizeof(type))
#define KLEN(type,arity) (nneighbors*arity*sizeof(type))
void init_dev_structures(int nparticles, int nneighbors) {
  //packed
  cudaMalloc((void **)&d_x,       NLEN(double,3));
  cudaMalloc((void **)&d_v,       NLEN(double,3));
  cudaMalloc((void **)&d_omega,   NLEN(double,3));
  cudaMalloc((void **)&d_radius,  NLEN(double,1));
  cudaMalloc((void **)&d_mass,    NLEN(double,1));
  cudaMalloc((void **)&d_type,    NLEN(int,   1));

  //unpacked(i)
  cudaMalloc((void **)&d_xi,      KLEN(double,3));
  cudaMalloc((void **)&d_vi,      KLEN(double,3));
  cudaMalloc((void **)&d_omegai,  KLEN(double,3));
  cudaMalloc((void **)&d_radiusi, KLEN(double,1));
  cudaMalloc((void **)&d_massi,   KLEN(double,1));
  cudaMalloc((void **)&d_typei,   KLEN(int   ,1));

  //unpacked(j)
  cudaMalloc((void **)&d_xj,      KLEN(double,3));
  cudaMalloc((void **)&d_vj,      KLEN(double,3));
  cudaMalloc((void **)&d_omegaj,  KLEN(double,3));
  cudaMalloc((void **)&d_radiusj, KLEN(double,1));
  cudaMalloc((void **)&d_massj,   KLEN(double,1));
  cudaMalloc((void **)&d_typej,   KLEN(int   ,1));

  //outputs
  cudaMalloc((void **)&d_force,   NLEN(double,3));
  cudaMalloc((void **)&d_torque,  NLEN(double,3));
  cudaMalloc((void **)&d_fdelta,  KLEN(double,3));
  cudaMalloc((void **)&d_tdeltai, KLEN(double,3));
  cudaMalloc((void **)&d_tdeltaj, KLEN(double,3));
}

void free_dev_structures() {
  //packed
  cudaFree(d_x);
  cudaFree(d_v);
  cudaFree(d_omega);
  cudaFree(d_radius);
  cudaFree(d_mass);
  cudaFree(d_type);

  //unpacked(i)
  cudaFree(d_xi);
  cudaFree(d_vi);
  cudaFree(d_omegai);
  cudaFree(d_radiusi);
  cudaFree(d_massi);
  cudaFree(d_typei);

  //unpacked(j)
  cudaFree(d_xj);
  cudaFree(d_vj);
  cudaFree(d_omegaj);
  cudaFree(d_radiusj);
  cudaFree(d_massj);
  cudaFree(d_typej);

  //outputs
  cudaFree(d_force);
  cudaFree(d_torque);
  cudaFree(d_fdelta);
  cudaFree(d_tdeltai);
  cudaFree(d_tdeltaj);
}

void run(struct params *input, int num_iter) {
  NeighListLike *nl = new NeighListLike(input);

  int block_size = BLOCK_SIZE;
  int nparticles = input->nnode;
  dim3 tpa_grid_size(
    min(nparticles/block_size, MAX_GRID_DIM),
    max((int)ceil(((float)nparticles/block_size)/MAX_GRID_DIM), 1));
  int nneighbors = nl->maxpage * nl->pgsize;
  dim3 tpn_grid_size(
    min(nneighbors/block_size, MAX_GRID_DIM),
    max((int)ceil(((float)nneighbors/block_size)/MAX_GRID_DIM), 1));
#if DEBUG
  printf("block_size = %d\n", block_size);
  printf("nparticles = %d\n", nparticles);
  printf("nneighbors = %d -> %d (maxpage=%d, pgsize=%d)\n",
    input->nedge, nneighbors, nl->maxpage, nl->pgsize);
  printf("tpa_grid   = { %d, %d, %d }\n",
    tpa_grid_size.x, tpa_grid_size.y, tpa_grid_size.z);
  printf("tpn_grid   = { %d, %d, %d }\n",
    tpn_grid_size.x, tpn_grid_size.y, tpn_grid_size.z);
#endif

  //ONE-TIME COSTS
  one_time.push_back(SimpleTimer("hertz_consts"));
  one_time.back().start();
  setup_hertz_constants(input);
  one_time.back().stop_and_add_to_total();
  no_cuda_error("hertz_constants");

  one_time.push_back(SimpleTimer("init_nl"));
  one_time.back().start();
  HertzCudaNeighList *d_nl = new HertzCudaNeighList(
    block_size,
    input->nnode,
    nl->maxpage, nl->pgsize);
  one_time.back().stop_and_add_to_total();
  no_cuda_error("init_nl");

  one_time.push_back(SimpleTimer("malloc"));
  one_time.back().start();
  init_dev_structures(nparticles, nneighbors);
  one_time.back().stop_and_add_to_total();
  no_cuda_error("init_dev_structures");

  one_time.push_back(SimpleTimer("memcpy"));
  one_time.back().start();
  cudaMemcpy(d_force,  input->force,  NLEN(double,3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_torque, input->torque, NLEN(double,3), cudaMemcpyHostToDevice);
  one_time.back().stop_and_add_to_total();
  no_cuda_error("memcpy");

  //NL-REFRESH COSTS
  nl_refresh.push_back(SimpleTimer("nl_reload"));
  nl_refresh.back().start();
  d_nl->reload(
    nl->numneigh,
    nl->firstneigh,
    nl->pages,
    nl->maxpage,
    nl->dpages,
    nl->tpages);
  nl_refresh.back().stop_and_add_to_total();
  no_cuda_error("nl_reload");

  nl_refresh.push_back(SimpleTimer("memcpy_unpack"));
  nl_refresh.back().start();
  cudaMemcpy(d_radius, input->radius, NLEN(double,1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mass,   input->mass,   NLEN(double,1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_type,   input->type,   NLEN(int,1),    cudaMemcpyHostToDevice);
  nl_refresh.back().stop_and_add_to_total();
  no_cuda_error("memcpy_unpack");

  nl_refresh.push_back(SimpleTimer("unpack_ro"));
  nl_refresh.back().start();
  unpack_ro_data<<<tpn_grid_size, block_size>>>(
    nneighbors,
              d_nl->d_valid,
              d_nl->d_dati,  d_nl->d_neighidx,
    d_radius, d_radiusi,     d_radiusj,
    d_mass,   d_massi,       d_massj,
    d_type,   d_typei,       d_typej
  );
  cudaThreadSynchronize();
  nl_refresh.back().stop_and_add_to_total();
  no_cuda_error("unpack_ro");

  // PER-ITER COSTS
  per_iter.push_back(SimpleTimer("memcpy_reload"));
  per_iter.push_back(SimpleTimer("unpack_reload"));
  per_iter.push_back(SimpleTimer("compute"));
  per_iter.push_back(SimpleTimer("collect"));
  per_iter.push_back(SimpleTimer("memcpy_results"));
  per_iter_timings.push_back(vector<double>(num_iter));
  per_iter_timings.push_back(vector<double>(num_iter));
  per_iter_timings.push_back(vector<double>(num_iter));
  per_iter_timings.push_back(vector<double>(num_iter));
  per_iter_timings.push_back(vector<double>(num_iter));

  double *force  = new double[nparticles*3];
  double *torque = new double[nparticles*3];
  for (int run=0; run<num_iter; run++) {
    //make copies
    cudaMemcpy(d_force,  input->force,  NLEN(double,3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_torque, input->torque, NLEN(double,3), cudaMemcpyHostToDevice);
    cudaMemset(d_fdelta,  0, KLEN(double,3));
    cudaMemset(d_tdeltai, 0, KLEN(double,3));
    cudaMemset(d_tdeltaj, 0, KLEN(double,3));
    nl->restore();
    d_nl->load_shear(nl->dpages);
    no_cuda_error("make_copies");

    end_to_end.start();

    per_iter[0].start();
    cudaMemcpy(d_x,     input->x,     NLEN(double,3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,     input->v,     NLEN(double,3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega, input->omega, NLEN(double,3), cudaMemcpyHostToDevice);
    double d0 = per_iter[0].stop_and_add_to_total();
    per_iter_timings[0][run] = d0;
    no_cuda_error("memcpy_reload");

    //TODO: check if realloc of unpacked ij data necessary
    per_iter[1].start();
    unpack_reload_data<<<tpn_grid_size, block_size>>>(
      nneighbors,
               d_nl->d_valid,
               d_nl->d_dati,  d_nl->d_neighidx,
      d_x,     d_xi,          d_xj,
      d_v,     d_vi,          d_vj,
      d_omega, d_omegai,      d_omegaj
    );
    cudaThreadSynchronize();
    double d1 = per_iter[1].stop_and_add_to_total();
    per_iter_timings[1][run] = d1;
    no_cuda_error("unpack_reload");

    per_iter[2].start();
#ifdef TRACE
    cudaPrintfInit();
#endif
    compute<<<tpn_grid_size, block_size>>>(
      nneighbors,
      d_nl->d_valid,
#ifdef TRACE
      d_nl->d_dati,  d_nl->d_neighidx,
#endif
      d_xi,          d_xj,
      d_vi,          d_vj,
      d_omegai,      d_omegaj,
      d_radiusi,     d_radiusj,
      d_massi,       d_massj,
      d_typei,       d_typej,
      //outputs
      d_fdelta,
      d_tdeltai,     d_tdeltaj,
      d_nl->d_shear
    );
    cudaThreadSynchronize();
    double d2 = per_iter[2].stop_and_add_to_total();
    per_iter_timings[2][run] = d2;
    no_cuda_error("compute");
#ifdef TRACE
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
#endif

    per_iter[3].start();
    collect<<<tpa_grid_size, block_size>>>(
      nparticles,
      d_fdelta,
      d_tdeltai, d_tdeltaj,
                        d_nl->d_offset, d_nl->d_numneigh,
#if HALFNL
      d_nl->d_tad,      d_nl->d_ffo,    d_nl->d_nel,
#endif
      d_force,
      d_torque);
    cudaThreadSynchronize();
    double d3 = per_iter[3].stop_and_add_to_total();
    per_iter_timings[3][run] = d3;
    no_cuda_error("collect");

    per_iter[4].start();
    cudaMemcpy(force,  d_force,  NLEN(double,3), cudaMemcpyDeviceToHost);
    cudaMemcpy(torque, d_torque, NLEN(double,3), cudaMemcpyDeviceToHost);
    d_nl->unload_shear(nl->dpages);
    double d4 = per_iter[4].stop_and_add_to_total();
    per_iter_timings[4][run] = d4;
    no_cuda_error("memcpy_results");

    double d5 = end_to_end.stop_and_add_to_total();
    end_to_end_timings.push_back(d5);

    check_result(input, nl, force, torque, nl->firstdouble,
      /*threshold=*/0.5,
      /*verbose=*/false,
      /*die_on_flag=*/true);
  }
  delete[] force;
  delete[] torque;
  free_dev_structures();
  no_cuda_error("free_dev_structures");
}

