/*
 * Neighbor list decomposition of the hertz pairwise kernel. 
 */

#define NSLOT 32
//#define KERNEL_PRINT    //< debug printing in kernel

#ifdef GPU_TIMER
  #include "cuda_timer.h"
#elif POSIX_TIMER
  #include "posix_timer.h"
#else
  #include "simple_timer.h"
#endif

#include "check_result_vector.h"
#include "cuda_common.h"
#include "framework.h"
#include "hertz_constants.h"
#include "particle.h"
#include <sstream>

using namespace std;

// --------------------------------------------------------------------------
// DEVICE KERNEL
// --------------------------------------------------------------------------

__device__ void pair_interaction(
  //inputs
    double *xi, double *xj,           //position
    double *vi, double *vj,           //velocity
    double *omegai, double *omegaj,   //rotational velocity
    double radi, double radj,         //radius
    double massi, double massj,       //mass
    int typei, int typej,             //type
  //inouts
    double *shear,
    double *torque,
    double *force) {

  // del is the vector from j to i
  double delx = xi[0] - xj[0];
  double dely = xi[1] - xj[1];
  double delz = xi[2] - xj[2];

  double rsq = delx*delx + dely*dely + delz*delz;
  double radsum = radi + radj;
  if (rsq >= radsum*radsum) {
    //unset non-touching atoms
    shear[0] = 0.0;
    shear[1] = 0.0;
    shear[2] = 0.0;
  } else {
    //distance between centres of atoms i and j
    //or, magnitude of del vector
    double r = sqrt(rsq);
    double rinv = 1.0/r;
    double rsqinv = 1.0/rsq;

    // relative translational velocity
    double vr1 = vi[0] - vj[0];
    double vr2 = vi[1] - vj[1];
    double vr3 = vi[2] - vj[2];

    // normal component
    double vnnr = vr1*delx + vr2*dely + vr3*delz;
    double vn1 = delx*vnnr * rsqinv;
    double vn2 = dely*vnnr * rsqinv;
    double vn3 = delz*vnnr * rsqinv;

    // tangential component
    double vt1 = vr1 - vn1;
    double vt2 = vr2 - vn2;
    double vt3 = vr3 - vn3;

    // relative rotational velocity
    double wr1 = (radi*omegai[0] + radj*omegaj[0]) * rinv;
    double wr2 = (radi*omegai[1] + radj*omegaj[1]) * rinv;
    double wr3 = (radi*omegai[2] + radj*omegaj[2]) * rinv;

    // normal forces = Hookian contact + normal velocity damping
    double meff = massi*massj/(massi+massj);
    //not-implemented: freeze_group_bit

    double deltan = radsum-r;

    //derive contact model parameters (inlined)
    //Yeff, Geff, betaeff, coeffFrict are lookup tables
    double reff = radi * radj / (radi + radj);
    double sqrtval = sqrt(reff * deltan);
    double Sn = 2.    * d_yeff * sqrtval;
    double St = 8.    * d_geff * sqrtval;
    double kn = 4./3. * d_yeff * sqrtval;
    double kt = St;
    double gamman=-2.*sqrtFiveOverSix*d_betaeff*sqrt(Sn*meff);
    double gammat=-2.*sqrtFiveOverSix*d_betaeff*sqrt(St*meff);
    double xmu=d_coeffFrict;
    //not-implemented if (dampflag == 0) gammat = 0;
    kn /= d_nktv2p;
    kt /= d_nktv2p;

    double damp = gamman*vnnr*rsqinv;
	  double ccel = kn*(radsum-r)*rinv - damp;

    //not-implemented cohesionflag

    // relative velocities
    double vtr1 = vt1 - (delz*wr2-dely*wr3);
    double vtr2 = vt2 - (delx*wr3-delz*wr1);
    double vtr3 = vt3 - (dely*wr1-delx*wr2);

    // shear history effects
    shear[0] += vtr1 * d_dt;
    shear[1] += vtr2 * d_dt;
    shear[2] += vtr3 * d_dt;

    // rotate shear displacements
    double rsht = shear[0]*delx + shear[1]*dely + shear[2]*delz;
    rsht *= rsqinv;

    shear[0] -= rsht*delx;
    shear[1] -= rsht*dely;
    shear[2] -= rsht*delz;

    // tangential forces = shear + tangential velocity damping
    double fs1 = - (kt*shear[0] + gammat*vtr1);
    double fs2 = - (kt*shear[1] + gammat*vtr2);
    double fs3 = - (kt*shear[2] + gammat*vtr3);

    // rescale frictional displacements and forces if needed
    double fs = sqrt(fs1*fs1 + fs2*fs2 + fs3*fs3);
    double fn = xmu * fabs(ccel*r);
    double shrmag = 0;
    if (fs > fn) {
      shrmag = sqrt(shear[0]*shear[0] +
                    shear[1]*shear[1] +
                    shear[2]*shear[2]);
      if (shrmag != 0.0) {
        shear[0] = (fn/fs) * (shear[0] + gammat*vtr1/kt) - gammat*vtr1/kt;
        shear[1] = (fn/fs) * (shear[1] + gammat*vtr2/kt) - gammat*vtr2/kt;
        shear[2] = (fn/fs) * (shear[2] + gammat*vtr3/kt) - gammat*vtr3/kt;
        fs1 *= fn/fs;
        fs2 *= fn/fs;
        fs3 *= fn/fs;
      } else {
        fs1 = fs2 = fs3 = 0.0;
      }
    }

    double fx = delx*ccel + fs1;
    double fy = dely*ccel + fs2;
    double fz = delz*ccel + fs3;

    double tor1 = rinv * (dely*fs3 - delz*fs2);
    double tor2 = rinv * (delz*fs1 - delx*fs3);
    double tor3 = rinv * (delx*fs2 - dely*fs1);

    // this is what we've been working up to!
    force[0] += fx;
    force[1] += fy;
    force[2] += fz;

    torque[0] -= radi*tor1;
    torque[1] -= radi*tor2;
    torque[2] -= radi*tor3;
  }
}

__global__ void compute_kernel_tpa(
  int nparticles,   
  struct particle *particle_aos,
  int *numneigh,
  struct particle *neigh,
  double3 *shear,
  double *force,
  double *torque) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < nparticles && numneigh[idx] > 0) {
    double xi[3]; double xj[3];
    double vi[3]; double vj[3];
    double omegai[3]; double omegaj[3];
    double radiusi; double radiusj;
    double massi; double massj;
    int    typei; int typej;
    double shearij[3] = {0,0,0};
    double force_deltai[3] = {0,0,0};
    double torque_deltai[3] = {0,0,0};

    xi[0]     = particle_aos[idx].x[0];
    xi[1]     = particle_aos[idx].x[1];
    xi[2]     = particle_aos[idx].x[2];
    vi[0]     = particle_aos[idx].v[0];
    vi[1]     = particle_aos[idx].v[1];
    vi[2]     = particle_aos[idx].v[2];
    omegai[0] = particle_aos[idx].omega[0];
    omegai[1] = particle_aos[idx].omega[1];
    omegai[2] = particle_aos[idx].omega[2];
    radiusi   = particle_aos[idx].radius;
    massi     = particle_aos[idx].mass;
    typei     = particle_aos[idx].type;

    for (int jj=0; jj<numneigh[idx]; jj++) {
      int neigh_idx = (idx*NSLOT)+jj;
      //int j   = neigh[neigh_idx].idx;
      xj[0]     = neigh[neigh_idx].x[0];
      xj[1]     = neigh[neigh_idx].x[1];
      xj[2]     = neigh[neigh_idx].x[2];
      vj[0]     = neigh[neigh_idx].v[0];
      vj[1]     = neigh[neigh_idx].v[1];
      vj[2]     = neigh[neigh_idx].v[2];
      omegaj[0] = neigh[neigh_idx].omega[0];
      omegaj[1] = neigh[neigh_idx].omega[1];
      omegaj[2] = neigh[neigh_idx].omega[2];
      radiusj   = neigh[neigh_idx].radius;
      massj     = neigh[neigh_idx].mass;
      typej     = neigh[neigh_idx].type;

      shearij[0] = shear[neigh_idx].x;
      shearij[1] = shear[neigh_idx].y;
      shearij[2] = shear[neigh_idx].z;

      pair_interaction(
        xi, xj,
        vi, vj,
        omegai, omegaj,
        radiusi, radiusj,
        massi, massj,
        typei, typej,
        shearij, torque_deltai, force_deltai);

      shear[neigh_idx].x = shearij[0];
      shear[neigh_idx].y = shearij[1];
      shear[neigh_idx].z = shearij[2];
    }
    force[(idx*3)  ] += force_deltai[0];
    force[(idx*3)+1] += force_deltai[1];
    force[(idx*3)+2] += force_deltai[2];

    torque[(idx*3)  ] += torque_deltai[0];
    torque[(idx*3)+1] += torque_deltai[1];
    torque[(idx*3)+2] += torque_deltai[2];
  }
}

// --------------------------------------------------------------------------
// RUN
// --------------------------------------------------------------------------

void insert_particle(struct params *input, struct particle *aos, int id, int n) {
  assert(n < input->nnode);
  aos[id].idx      = n;
  aos[id].x[0]     = input->x[(n*3)  ];
  aos[id].x[1]     = input->x[(n*3)+1];
  aos[id].x[2]     = input->x[(n*3)+2];
  aos[id].v[0]     = input->v[(n*3)  ];
  aos[id].v[1]     = input->v[(n*3)+1];
  aos[id].v[2]     = input->v[(n*3)+2];
  aos[id].omega[0] = input->omega[(n*3)  ];
  aos[id].omega[1] = input->omega[(n*3)+1];
  aos[id].omega[2] = input->omega[(n*3)+2];
  aos[id].radius   = input->radius[n];
  aos[id].mass     = input->mass[n];
  aos[id].type     = input->type[n];
}

void build_particle_aos(struct params *input, struct particle *&d_particle_aos) {
  struct particle *aos = new particle[input->nnode];
  for (int n=0; n<input->nnode; n++) {
    insert_particle(input, aos, n, n);
  }
  const int aos_size = input->nnode*sizeof(struct particle);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_particle_aos, aos_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_particle_aos, aos, aos_size, cudaMemcpyHostToDevice));
}

void build_neighbor_list(
  int nslot,
  struct params *input, 
  int *&d_numneigh, struct particle *&d_neigh, double3 *&d_shear) {

  int *numneigh = new int[input->nnode*nslot];
  struct particle *neigh = new particle[input->nnode*nslot];
  double3 *shear = new double3[input->nnode*nslot];

  for (int i=0; i<input->nnode*nslot; i++) {
    numneigh[i] = 0;
  }
  for (int e=0; e<input->nedge; e++) {
    int i = input->edge[(e*2)  ];
    int j = input->edge[(e*2)+1];

    assert(numneigh[i] < nslot);
    int idx = (i*nslot) + numneigh[i];
    insert_particle(input, neigh, idx, j);
    shear[idx].x = input->shear[(e*3)  ];
    shear[idx].y = input->shear[(e*3)+1];
    shear[idx].z = input->shear[(e*3)+2];
    numneigh[i]++;

#ifndef NEWTON_THIRD
    assert(numneigh[j] < nslot);
    idx = (j*nslot) + numneigh[j];
    insert_particle(input, neigh, idx, i);
    shear[idx].x = input->shear[(e*3)  ];
    shear[idx].y = input->shear[(e*3)+1];
    shear[idx].z = input->shear[(e*3)+2];
    numneigh[j]++;
#endif
  }

  const int numneigh_size = input->nnode*nslot*sizeof(int);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_numneigh, numneigh_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_numneigh, numneigh, numneigh_size, cudaMemcpyHostToDevice));

  const int neigh_size = input->nnode*nslot*sizeof(struct particle);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_neigh, neigh_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_neigh, neigh, neigh_size, cudaMemcpyHostToDevice));

  const int shear_size = input->nnode*nslot*sizeof(double3);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_shear, shear_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_shear, shear, shear_size, cudaMemcpyHostToDevice));
}

void run(struct params *input, int num_iter) {
  one_time.push_back(SimpleTimer("hertz_constants"));
  one_time.back().start();
  setup_hertz_constants();
  one_time.back().stop_and_add_to_total();

  one_time.push_back(SimpleTimer("build_particle_aos"));
  struct particle *d_particle_aos = NULL;
  one_time.back().start();
  build_particle_aos(input, d_particle_aos);
  one_time.back().stop_and_add_to_total();
  assert(d_particle_aos);

  one_time.push_back(SimpleTimer("build_neigh_list"));
  int *d_numneigh = NULL;
  struct particle *d_neigh = NULL;
  double3 *d_shear = NULL;
  one_time.back().start();
  build_neighbor_list(NSLOT, input, d_numneigh, d_neigh, d_shear);
  one_time.back().stop_and_add_to_total();
  assert(d_numneigh);
  assert(d_neigh);
  assert(d_shear);

  one_time.push_back(SimpleTimer("malloc_force_torque"));
  one_time.back().start();
  double *d_force;
  const int force_size = input->nnode * 3 * sizeof(double);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_force, force_size));

  double *d_torque;
  const int torque_size = input->nnode * 3 * sizeof(double);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_torque, torque_size));
  one_time.back().stop_and_add_to_total();

  per_iter.push_back(SimpleTimer("compute_kernel"));
  for (int run=0; run<num_iter; run++) {
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_force, input->force, force_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_torque, input->torque, torque_size, cudaMemcpyHostToDevice));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Pre-compute-kernel error: %s.\n", cudaGetErrorString(err));
      exit(1);
    }
    const int blockSize = 128;
    dim3 gridSize((input->nnode / blockSize)+1);
    per_iter[0].start();
    compute_kernel_tpa<<<gridSize, blockSize>>>(
      input->nnode, d_particle_aos, d_numneigh, d_neigh, 
      d_shear, d_force, d_torque);
    cudaThreadSynchronize();
    per_iter[0].stop_and_add_to_total();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Post-compute-kernel error: %s.\n", cudaGetErrorString(err));
      exit(1);
    }

#if 0
    double *force_result = new double[input->nnode*3];
    double *torque_result = new double[input->nnode*3];
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(force_result, d_force, force_size, cudaMemcpyDeviceToHost));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(torque_result, d_torque, torque_size, cudaMemcpyDeviceToHost));
    for (int n=0; n<input->nnode; n++) {
      const double epsilon=0.00001;

      std::stringstream out;
      out << "force[" << n << "]";
      check_result_vector(
          out.str().c_str(),
          &input->expected_force[(n*3)], &force_result[(n*3)], epsilon, false, false);
      out.str("");

      out << "torque[" << n << "]";
      check_result_vector(
          out.str().c_str(),
          &input->expected_torque[(n*3)], &torque_result[(n*3)], epsilon, false, false);
    }
#endif
  }

  cudaFree(d_particle_aos);
  cudaFree(d_numneigh);
  cudaFree(d_neigh);
  cudaFree(d_shear);
  cudaFree(d_force);
  cudaFree(d_torque);
}
