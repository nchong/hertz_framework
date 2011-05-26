/*
 * Neighbor list decomposition of the hertz pairwise kernel.
 */

#define NSLOT 32          //< maximum number of neighbors per particle
//#define AOS_LAYOUT      //< AoS or SoA layout for read-only particle data
//#define KERNEL_PRINT    //< debug printing in kernel
//#define MAP_BUILD_CHECK //< bounds and sanity checking in build_inverse_map
//#define NEWTON_THIRD    //< use Newton's third law to halve computation
//#define COMPUTE_TPA     //< thread-per-atom decomposition
//#define PINNED_MEM      //< use pinned-memory for kernel output

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
#include "inverse_map.h"
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
    double *force
#ifdef NEWTON_THIRD
    , double *torquej
#endif
  ) {

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

#ifdef NEWTON_THIRD
    torquej[0] -= radj*tor1;
    torquej[1] -= radj*tor2;
    torquej[2] -= radj*tor3;
#endif
  }
}

__global__ void compute_kernel_tpa(
  int nparticles,
#ifdef AOS_LAYOUT
  struct particle *particle_aos,
#else
  struct particle particle_soa,
#endif
  int *numneigh,
#ifdef AOS_LAYOUT
  struct particle *neigh,
#else
  struct particle neigh,
#endif
  double3 *shear,
  double *force,
  double *torque
#ifdef NEWTON_THIRD
  ,
  double3 *fdelta,
  double3 *tdeltaj
#endif
  ) {

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

#ifdef AOS_LAYOUT
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
#else
    xi[0]     = particle_soa.x[(idx*3)+0];
    xi[1]     = particle_soa.x[(idx*3)+1];
    xi[2]     = particle_soa.x[(idx*3)+2];
    vi[0]     = particle_soa.v[(idx*3)+0];
    vi[1]     = particle_soa.v[(idx*3)+1];
    vi[2]     = particle_soa.v[(idx*3)+2];
    omegai[0] = particle_soa.omega[(idx*3)+0];
    omegai[1] = particle_soa.omega[(idx*3)+1];
    omegai[2] = particle_soa.omega[(idx*3)+2];
    radiusi   = particle_soa.radius[idx];
    massi     = particle_soa.mass[idx];
    typei     = particle_soa.type[idx];
#endif

    for (int jj=0; jj<numneigh[idx]; jj++) {
      int neigh_idx = (idx*NSLOT)+jj;
#ifdef AOS_LAYOUT
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
#else
      //int j   = neigh.idx[neigh_idx];
      xj[0]     = neigh.x[(neigh_idx*3)+0];
      xj[1]     = neigh.x[(neigh_idx*3)+1];
      xj[2]     = neigh.x[(neigh_idx*3)+2];
      vj[0]     = neigh.v[(neigh_idx*3)+0];
      vj[1]     = neigh.v[(neigh_idx*3)+1];
      vj[2]     = neigh.v[(neigh_idx*3)+2];
      omegaj[0] = neigh.omega[(neigh_idx*3)+0];
      omegaj[1] = neigh.omega[(neigh_idx*3)+1];
      omegaj[2] = neigh.omega[(neigh_idx*3)+2];
      radiusj   = neigh.radius[neigh_idx];
      massj     = neigh.mass[neigh_idx];
      typej     = neigh.type[neigh_idx];
#endif

      shearij[0] = shear[neigh_idx].x;
      shearij[1] = shear[neigh_idx].y;
      shearij[2] = shear[neigh_idx].z;

#ifdef NEWTON_THIRD
      double torque_deltaj[3] = {0,0,0};
      pair_interaction(
        xi, xj,
        vi, vj,
        omegai, omegaj,
        radiusi, radiusj,
        massi, massj,
        typei, typej,
        shearij, torque_deltai, force_deltai, torque_deltaj);

      fdelta[neigh_idx].x = force_deltai[0];
      fdelta[neigh_idx].y = force_deltai[1];
      fdelta[neigh_idx].z = force_deltai[2];

      tdeltaj[neigh_idx].x = torque_deltaj[0];
      tdeltaj[neigh_idx].y = torque_deltaj[1];
      tdeltaj[neigh_idx].z = torque_deltaj[2];
#else
      pair_interaction(
        xi, xj,
        vi, vj,
        omegai, omegaj,
        radiusi, radiusj,
        massi, massj,
        typei, typej,
        shearij, torque_deltai, force_deltai);
#endif

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

//NB: expects blockDim.x >= NSLOT
//    this kernel will produce incorrect results if blockDim.x < NSLOT
__global__ void compute_kernel_bpa(
  int nparticles,
#ifdef AOS_LAYOUT
  struct particle *particle_aos,
#else
  struct particle particle_soa,
#endif
  int *numneigh,
#ifdef AOS_LAYOUT
  struct particle *neigh,
#else
  struct particle neigh,
#endif
  double3 *shear,
  double *force,
  double *torque
#ifdef NEWTON_THIRD
  ,
  double3 *fdelta,
  double3 *tdeltaj
#endif
  ) {

  __shared__ double ftmp[NSLOT*3];
  __shared__ double ttmp[NSLOT*3];

  int jj = threadIdx.x;
  int idx = blockIdx.x;
  if (idx < nparticles && jj < numneigh[idx]) {
    double xi[3]; double xj[3];
    double vi[3]; double vj[3];
    double omegai[3]; double omegaj[3];
    double radiusi; double radiusj;
    double massi; double massj;
    int    typei; int typej;
    double shearij[3] = {0,0,0};
    double force_deltai[3] = {0,0,0};
    double torque_deltai[3] = {0,0,0};

#ifdef AOS_LAYOUT
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
#else
    xi[0]     = particle_soa.x[(idx*3)+0];
    xi[1]     = particle_soa.x[(idx*3)+1];
    xi[2]     = particle_soa.x[(idx*3)+2];
    vi[0]     = particle_soa.v[(idx*3)+0];
    vi[1]     = particle_soa.v[(idx*3)+1];
    vi[2]     = particle_soa.v[(idx*3)+2];
    omegai[0] = particle_soa.omega[(idx*3)+0];
    omegai[1] = particle_soa.omega[(idx*3)+1];
    omegai[2] = particle_soa.omega[(idx*3)+2];
    radiusi   = particle_soa.radius[idx];
    massi     = particle_soa.mass[idx];
    typei     = particle_soa.type[idx];
#endif

    int neigh_idx = (idx*NSLOT)+jj;
#ifdef AOS_LAYOUT
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
#else
    //int j   = neigh.idx[neigh_idx];
    xj[0]     = neigh.x[(neigh_idx*3)+0];
    xj[1]     = neigh.x[(neigh_idx*3)+1];
    xj[2]     = neigh.x[(neigh_idx*3)+2];
    vj[0]     = neigh.v[(neigh_idx*3)+0];
    vj[1]     = neigh.v[(neigh_idx*3)+1];
    vj[2]     = neigh.v[(neigh_idx*3)+2];
    omegaj[0] = neigh.omega[(neigh_idx*3)+0];
    omegaj[1] = neigh.omega[(neigh_idx*3)+1];
    omegaj[2] = neigh.omega[(neigh_idx*3)+2];
    radiusj   = neigh.radius[neigh_idx];
    massj     = neigh.mass[neigh_idx];
    typej     = neigh.type[neigh_idx];
#endif

    shearij[0] = shear[neigh_idx].x;
    shearij[1] = shear[neigh_idx].y;
    shearij[2] = shear[neigh_idx].z;

#ifdef NEWTON_THIRD
    double torque_deltaj[3] = {0,0,0};
    pair_interaction(
      xi, xj,
      vi, vj,
      omegai, omegaj,
      radiusi, radiusj,
      massi, massj,
      typei, typej,
      shearij, torque_deltai, force_deltai, torque_deltaj);

    fdelta[neigh_idx].x = force_deltai[0];
    fdelta[neigh_idx].y = force_deltai[1];
    fdelta[neigh_idx].z = force_deltai[2];

    tdeltaj[neigh_idx].x = torque_deltaj[0];
    tdeltaj[neigh_idx].y = torque_deltaj[1];
    tdeltaj[neigh_idx].z = torque_deltaj[2];
#else
    pair_interaction(
      xi, xj,
      vi, vj,
      omegai, omegaj,
      radiusi, radiusj,
      massi, massj,
      typei, typej,
      shearij, torque_deltai, force_deltai);
#endif

    shear[neigh_idx].x = shearij[0];
    shear[neigh_idx].y = shearij[1];
    shear[neigh_idx].z = shearij[2];

    ftmp[(jj*3)  ] = force_deltai[0];
    ftmp[(jj*3)+1] = force_deltai[1];
    ftmp[(jj*3)+2] = force_deltai[2];

    ttmp[(jj*3)  ] = torque_deltai[0];
    ttmp[(jj*3)+1] = torque_deltai[1];
    ttmp[(jj*3)+2] = torque_deltai[2];

    __syncthreads();

    //reduce
    if (jj == 0) {
      //serial reduce
      for (int i=1; i<numneigh[idx]; i++) {
        force_deltai[0] += ftmp[(i*3)  ];
        force_deltai[1] += ftmp[(i*3)+1];
        force_deltai[2] += ftmp[(i*3)+2];

        torque_deltai[0] += ttmp[(i*3)  ];
        torque_deltai[1] += ttmp[(i*3)+1];
        torque_deltai[2] += ttmp[(i*3)+2];
      }

      force[(idx*3)  ] += force_deltai[0];
      force[(idx*3)+1] += force_deltai[1];
      force[(idx*3)+2] += force_deltai[2];

      torque[(idx*3)  ] += torque_deltai[0];
      torque[(idx*3)+1] += torque_deltai[1];
      torque[(idx*3)+2] += torque_deltai[2];
    }
  }
}

__global__ void gather_kernel(
  int nparticles,
  double3 *force_delta, double3 *torquej_delta,
  int *joffset, int *jcount, int *jmapinv,
  //outputs
  double *force, double *torque) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < nparticles) {
    double fdelta[3] = {0.0, 0.0, 0.0};
    double tdelta[3] = {0.0, 0.0, 0.0};

    int joff = joffset[idx];
    for (int i=0; i<jcount[idx]; i++) {
      int e = jmapinv[joff+i];

      fdelta[0] -= force_delta[e].x;
      fdelta[1] -= force_delta[e].y;
      fdelta[2] -= force_delta[e].z;

      tdelta[0] += torquej_delta[e].x;
      tdelta[1] += torquej_delta[e].y;
      tdelta[2] += torquej_delta[e].z;
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

void insert_particle(struct params *input, 
#ifdef AOS_LAYOUT
  struct particle *particle_aos, 
#else
  struct particle particle_soa, 
#endif
  int id, int n) {
  assert(n < input->nnode);
#ifdef AOS_LAYOUT
  particle_aos[id].idx      = n;
  particle_aos[id].x[0]     = input->x[(n*3)  ];
  particle_aos[id].x[1]     = input->x[(n*3)+1];
  particle_aos[id].x[2]     = input->x[(n*3)+2];
  particle_aos[id].v[0]     = input->v[(n*3)  ];
  particle_aos[id].v[1]     = input->v[(n*3)+1];
  particle_aos[id].v[2]     = input->v[(n*3)+2];
  particle_aos[id].omega[0] = input->omega[(n*3)  ];
  particle_aos[id].omega[1] = input->omega[(n*3)+1];
  particle_aos[id].omega[2] = input->omega[(n*3)+2];
  particle_aos[id].radius   = input->radius[n];
  particle_aos[id].mass     = input->mass[n];
  particle_aos[id].type     = input->type[n];
#else
  particle_soa.idx[id]         = n;
  particle_soa.x[(id*3)+0]     = input->x[(n*3)  ];
  particle_soa.x[(id*3)+1]     = input->x[(n*3)+1];
  particle_soa.x[(id*3)+2]     = input->x[(n*3)+2];
  particle_soa.v[(id*3)+0]     = input->v[(n*3)  ];
  particle_soa.v[(id*3)+1]     = input->v[(n*3)+1];
  particle_soa.v[(id*3)+2]     = input->v[(n*3)+2];
  particle_soa.omega[(id*3)+0] = input->omega[(n*3)  ];
  particle_soa.omega[(id*3)+1] = input->omega[(n*3)+1];
  particle_soa.omega[(id*3)+2] = input->omega[(n*3)+2];
  particle_soa.radius[id]      = input->radius[n];
  particle_soa.mass[id]        = input->mass[n];
  particle_soa.type[id]        = input->type[n];
#endif
}

void build_particle_list(struct params *input, 
#ifdef AOS_LAYOUT 
    struct particle *&d_particle_aos
#else
    struct particle &d_particle_soa
#endif 
) {
#ifdef AOS_LAYOUT
  struct particle *particle_aos = new particle[input->nnode];
  for (int n=0; n<input->nnode; n++) {
    insert_particle(input, particle_aos, n, n);
  }
  const int aos_size = input->nnode*sizeof(struct particle);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_particle_aos, aos_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_particle_aos, particle_aos, aos_size, cudaMemcpyHostToDevice));
#else
  struct particle particle_soa;
  particle_soa.idx    = new int[input->nnode];
  particle_soa.x      = new double[input->nnode*3];
  particle_soa.v      = new double[input->nnode*3];
  particle_soa.omega  = new double[input->nnode*3];
  particle_soa.radius = new double[input->nnode];
  particle_soa.mass   = new double[input->nnode];
  particle_soa.type   = new int[input->nnode];
  for (int n=0; n<input->nnode; n++) {
    insert_particle(input, particle_soa, n, n);
  }
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_particle_soa.idx, input->nnode*sizeof(int)));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_particle_soa.x, input->nnode*3*sizeof(double)));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_particle_soa.v, input->nnode*3*sizeof(double)));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_particle_soa.omega, input->nnode*3*sizeof(double)));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_particle_soa.radius, input->nnode*sizeof(double)));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_particle_soa.mass, input->nnode*sizeof(double)));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_particle_soa.type, input->nnode*sizeof(int)));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_particle_soa.idx, particle_soa.idx, input->nnode*sizeof(int), cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_particle_soa.x, particle_soa.x, input->nnode*3*sizeof(double), cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_particle_soa.v, particle_soa.v, input->nnode*3*sizeof(double), cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_particle_soa.omega, particle_soa.omega, input->nnode*3*sizeof(double), cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_particle_soa.radius, particle_soa.radius, input->nnode*sizeof(double), cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_particle_soa.mass, particle_soa.mass, input->nnode*sizeof(double), cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_particle_soa.type, particle_soa.type, input->nnode*sizeof(int), cudaMemcpyHostToDevice));
#endif
}

void build_neighbor_list(
  int nslot,
  struct params *input,
  int *&d_numneigh, 
#ifdef AOS_LAYOUT
  struct particle *&d_neigh,
#else
  struct particle &d_neigh,
#endif
  double3 *&d_shear
#ifdef NEWTON_THIRD
  ,
  int &delta_size, double3 *&d_fdelta, double3 *&d_tdeltaj,
  int *&d_joffset, int *&d_jcount, int *&d_jmapinv
#endif
  ) {

  //numneigh[n]
  //is the number of neighbors for particle n
  int *numneigh = new int[input->nnode*nslot];
  //neigh[(n*nslot)+i]
  //is the struct for the i-th particle in contact with particle n
#ifdef AOS_LAYOUT
  struct particle *neigh = new particle[input->nnode*nslot];
#else
  struct particle neigh;
  neigh.idx    = new int[input->nnode*nslot];
  neigh.x      = new double[input->nnode*nslot*3];
  neigh.v      = new double[input->nnode*nslot*3];
  neigh.omega  = new double[input->nnode*nslot*3];
  neigh.radius = new double[input->nnode*nslot];
  neigh.mass   = new double[input->nnode*nslot];
  neigh.type   = new int[input->nnode*nslot];
#endif
  //shear[(n*nslot)+i]
  //is the shear for the i-th particle in contact with particle n
  double3 *shear = new double3[input->nnode*nslot];

  for (int i=0; i<input->nnode*nslot; i++) {
    numneigh[i] = 0;
  }
  for (int e=0; e<input->nedge; e++) {
    int n1 = input->edge[(e*2)  ];
    int n2 = input->edge[(e*2)+1];

    assert(numneigh[n1] < nslot);
    int idx = (n1*nslot) + numneigh[n1];
    insert_particle(input, neigh, idx, n2);
    shear[idx].x = input->shear[(e*3)  ];
    shear[idx].y = input->shear[(e*3)+1];
    shear[idx].z = input->shear[(e*3)+2];
    numneigh[n1]++;

#ifndef NEWTON_THIRD
    //insert the symmetric contact if not using Newton's Third Law
    assert(numneigh[n2] < nslot);
    idx = (n2*nslot) + numneigh[n2];
    insert_particle(input, neigh, idx, n1);
    shear[idx].x = input->shear[(e*3)  ];
    shear[idx].y = input->shear[(e*3)+1];
    shear[idx].z = input->shear[(e*3)+2];
    numneigh[n2]++;
#endif
  }

  const int numneigh_size = input->nnode*nslot*sizeof(int);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_numneigh, numneigh_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_numneigh, numneigh, numneigh_size, cudaMemcpyHostToDevice));

#ifdef AOS_LAYOUT
  const int neigh_size = input->nnode*nslot*sizeof(struct particle);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_neigh, neigh_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_neigh, neigh, neigh_size, cudaMemcpyHostToDevice));
#else
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_neigh.idx, input->nnode*nslot*sizeof(int)));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_neigh.x, input->nnode*nslot*3*sizeof(double)));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_neigh.v, input->nnode*nslot*3*sizeof(double)));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_neigh.omega, input->nnode*nslot*3*sizeof(double)));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_neigh.radius, input->nnode*nslot*sizeof(double)));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_neigh.mass, input->nnode*nslot*sizeof(double)));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_neigh.type, input->nnode*nslot*sizeof(int)));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_neigh.idx, neigh.idx, input->nnode*nslot*sizeof(int), cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_neigh.x, neigh.x, input->nnode*3*nslot*sizeof(double), cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_neigh.v, neigh.v, input->nnode*3*nslot*sizeof(double), cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_neigh.omega, neigh.omega, input->nnode*3*nslot*sizeof(double), cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_neigh.radius, neigh.radius, input->nnode*nslot*sizeof(double), cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_neigh.mass, neigh.mass, input->nnode*nslot*sizeof(double), cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_neigh.type, neigh.type, input->nnode*nslot*sizeof(int), cudaMemcpyHostToDevice));
#endif

  const int shear_size = input->nnode*nslot*sizeof(double3);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_shear, shear_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_shear, shear, shear_size, cudaMemcpyHostToDevice));

#ifdef NEWTON_THIRD
  //jmap[(n*nslot)+i]
  //is the idx of the i-th particle in contact with particle n
  int *jmap = new int[input->nnode*nslot];
  for (int i=0; i<input->nnode*nslot; i++) {
    jmap[i] = neigh[i].idx;
  }

  //build an inverse mapping of jmap
  int *joffset;
  int *jcount;
  int *jmapinv;
  build_inverse_map(
    numneigh, jmap, input->nnode/*T*/, nslot, input->nnode/*K*/,
    joffset, jcount, jmapinv);

  const int joffset_jcount_size = input->nnode*sizeof(int);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_joffset, joffset_jcount_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_joffset, joffset, joffset_jcount_size, cudaMemcpyHostToDevice));

  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_jcount, joffset_jcount_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_jcount, jcount, joffset_jcount_size, cudaMemcpyHostToDevice));

  const int jmapinv_size = input->nnode*nslot*sizeof(int);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_jmapinv, jmapinv_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_jmapinv, jmapinv, jmapinv_size, cudaMemcpyHostToDevice));

  delta_size = input->nnode*nslot*sizeof(double3);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_fdelta, delta_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_tdeltaj, delta_size));
#endif

}

// --------------------------------------------------------------------------
// RUN
// --------------------------------------------------------------------------

void run(struct params *input, int num_iter) {

  //--------------------
  // One-time only costs
  //--------------------
  one_time.push_back(SimpleTimer("hertz_constants"));
  one_time.back().start();
  setup_hertz_constants();
  one_time.back().stop_and_add_to_total();

  one_time.push_back(SimpleTimer("build_particle_list"));
#ifdef AOS_LAYOUT
  struct particle *d_particle_list = NULL;
#else
  struct particle d_particle_list;
#endif
  one_time.back().start();
  build_particle_list(input, d_particle_list);
  one_time.back().stop_and_add_to_total();
#ifdef AOS_LAYOUT
  assert(d_particle_list);
#endif

  one_time.push_back(SimpleTimer("build_neigh_list"));
  int *d_numneigh = NULL;
#ifdef AOS_LAYOUT
  struct particle *d_neigh = NULL;
#else
  struct particle d_neigh;
#endif
  double3 *d_shear = NULL;
#ifdef NEWTON_THIRD
  int delta_size;
  double3 *d_fdelta = NULL;
  double3 *d_tdeltaj = NULL;
  int *d_joffset = NULL;
  int *d_jcount = NULL;
  int *d_jmapinv = NULL;
  one_time.back().start();
  build_neighbor_list(NSLOT, input, d_numneigh, d_neigh, d_shear,
    delta_size, d_fdelta, d_tdeltaj,
    d_joffset, d_jcount, d_jmapinv);
  one_time.back().stop_and_add_to_total();
  assert(d_fdelta);
  assert(d_tdeltaj);
  assert(d_joffset);
  assert(d_jcount);
  assert(d_jmapinv);
#else
  one_time.back().start();
  build_neighbor_list(NSLOT, input, d_numneigh, d_neigh, d_shear);
  one_time.back().stop_and_add_to_total();
#endif
  assert(d_numneigh);
#ifdef AOS_LAYOUT
  assert(d_neigh);
#endif
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

  //TODO(1): copy real x, v, omega in PREPROCESS
  //These are dummy structures just for timing
  const int d_x_size = input->nnode * 3 * sizeof(double);
  double *d_fake_x;
  double *d_fake_v;
  double *d_fake_omega;
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_fake_x, d_x_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_fake_v, d_x_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_fake_omega, d_x_size));

#ifdef PINNED_MEM
  one_time.push_back(SimpleTimer("pinned_mem"));
  one_time.back().start();
  double *h_x;
  double *h_v;
  double *h_omega;
  double *h_force;
  double *h_torque;
  ASSERT_NO_CUDA_ERROR(
    cudaMallocHost((void **)&h_x, d_x_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMallocHost((void **)&h_v, d_x_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMallocHost((void **)&h_omega, d_x_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMallocHost((void **)&h_force, force_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMallocHost((void **)&h_torque, torque_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(h_x, input->x, d_x_size, cudaMemcpyHostToHost));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(h_v, input->v, d_x_size, cudaMemcpyHostToHost));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(h_omega, input->omega, d_x_size, cudaMemcpyHostToHost));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(h_force, input->force, force_size, cudaMemcpyHostToHost));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(h_torque, input->torque, torque_size, cudaMemcpyHostToHost));
  one_time.back().stop_and_add_to_total();
#endif

  //--------------------
  // Per-iteration costs
  //--------------------

  per_iter.push_back(SimpleTimer("memcpy_to_dev"));
#ifdef COMPUTE_TPA
  per_iter.push_back(SimpleTimer("compute_kernel_tpa"));
#else
  per_iter.push_back(SimpleTimer("compute_kernel_bpa"));
#endif
  per_iter.push_back(SimpleTimer("gather_kernel"));
  per_iter.push_back(SimpleTimer("result_fetch"));

#ifdef PINNED_MEM
  double3 *shear_result;
  double *force_result;
  double *torque_result;
  ASSERT_NO_CUDA_ERROR(
    cudaMallocHost((void **)&shear_result, input->nnode*NSLOT*sizeof(double3)));
  ASSERT_NO_CUDA_ERROR(
    cudaMallocHost((void **)&force_result, input->nnode*3*sizeof(double)));
  ASSERT_NO_CUDA_ERROR(
    cudaMallocHost((void **)&torque_result, input->nnode*3*sizeof(double)));
#else
  double3 *shear_result = new double3[input->nnode*NSLOT];
  double *force_result = new double[input->nnode*3];
  double *torque_result = new double[input->nnode*3];
#endif

  for (int run=0; run<num_iter; run++) {
    //PREPROCESSING
    //copy across structures that change between kernel invocations,
    //reset delta structures (force/torque)
    //TODO(1): just copy dummy structures for timing
    per_iter[0].start();
#ifdef PINNED_MEM
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_fake_x, h_x, d_x_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_fake_v, h_v, d_x_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_fake_omega, h_omega, d_x_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_force, h_force, force_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_torque, h_torque, torque_size, cudaMemcpyHostToDevice));
#else
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_fake_x, input->x, d_x_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_fake_v, input->v, d_x_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_fake_omega, input->omega, d_x_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_force, input->force, force_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_torque, input->torque, torque_size, cudaMemcpyHostToDevice));
#endif
#ifdef NEWTON_THIRD
    ASSERT_NO_CUDA_ERROR(
      cudaMemset((void *)d_fdelta, 0, delta_size));
    ASSERT_NO_CUDA_ERROR(
      cudaMemset((void *)d_tdeltaj, 0, delta_size));
#endif
    per_iter[0].stop_and_add_to_total();

    //-----------------------------------------------------------------------

    //KERNEL INVOCATION
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Pre-compute-kernel error: %s.\n", cudaGetErrorString(err));
      exit(1);
    }

#ifdef KERNEL_PRINT
    cudaPrintfInit();
#endif
    per_iter[1].start();
#ifdef NEWTON_THIRD
  #ifdef COMPUTE_TPA
    const int blockSize = 128;
    dim3 gridSize((input->nnode / blockSize)+1);
    compute_kernel_tpa<<<gridSize, blockSize>>>(
      input->nnode, d_particle_list, d_numneigh, d_neigh,
      d_shear, d_force, d_torque, d_fdelta, d_tdeltaj);
  #else //COMPUTE_BPA
    const int blockSize = NSLOT;
    dim3 gridSize(input->nnode);
    compute_kernel_bpa<<<gridSize, blockSize>>>(
      input->nnode, d_particle_list, d_numneigh, d_neigh,
      d_shear, d_force, d_torque, d_fdelta, d_tdeltaj);
  #endif
#else
  #ifdef COMPUTE_TPA
    const int blockSize = 128;
    dim3 gridSize((input->nnode / blockSize)+1);
    compute_kernel_tpa<<<gridSize, blockSize>>>(
      input->nnode, d_particle_list, d_numneigh, d_neigh,
      d_shear, d_force, d_torque);
  #else //COMPUTE_BPA
    const int blockSize = NSLOT;
    dim3 gridSize(input->nnode);
    compute_kernel_bpa<<<gridSize, blockSize>>>(
      input->nnode, d_particle_list, d_numneigh, d_neigh,
      d_shear, d_force, d_torque);
  #endif
#endif
    cudaThreadSynchronize();
    per_iter[1].stop_and_add_to_total();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Post-compute-kernel error: %s.\n", cudaGetErrorString(err));
      exit(1);
    }

#ifdef NEWTON_THIRD
    const int gatherBlockSize = 128;
    dim3 gatherGridSize((input->nnode / gatherBlockSize)+1);
    per_iter[2].start();
    gather_kernel<<<gatherGridSize, gatherBlockSize>>>(
      input->nnode,
      d_fdelta, d_tdeltaj,
      d_joffset, d_jcount, d_jmapinv,
      d_force, d_torque);
    cudaThreadSynchronize();
    per_iter[2].stop_and_add_to_total();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Post-gather error: %s.\n", cudaGetErrorString(err));
      exit(1);
    }
#endif

#ifdef KERNEL_PRINT
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
#endif

    //-----------------------------------------------------------------------

    //POSTPROCESSING
    //memcpy data back to host
    const int shear_size = input->nnode*NSLOT*sizeof(double3);
    per_iter[3].start();
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(shear_result, d_shear, shear_size, cudaMemcpyDeviceToHost));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(force_result, d_force, force_size, cudaMemcpyDeviceToHost));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(torque_result, d_torque, torque_size, cudaMemcpyDeviceToHost));
    per_iter[3].stop_and_add_to_total();

#if 0
    if (run == 0) {
      for (int n=0; n<input->nnode; n++) {
        std::stringstream out;
        out << "force[" << n << "]";
        check_result_vector(
            out.str().c_str(),
            &input->expected_force[(n*3)], &force_result[(n*3)]);
        out.str("");

        out << "torque[" << n << "]";
        check_result_vector(
            out.str().c_str(),
            &input->expected_torque[(n*3)], &torque_result[(n*3)]);
      }
      //TODO check shear
    }
#endif
  }

#ifdef PINNED_MEM
  cudaFreeHost(h_x);
  cudaFreeHost(h_v);
  cudaFreeHost(h_omega);
  cudaFreeHost(h_force);
  cudaFreeHost(h_torque);
  cudaFreeHost(shear_result);
  cudaFreeHost(force_result);
  cudaFreeHost(torque_result);
#endif
#ifdef NEWTON_THIRD
  cudaFree(d_joffset);
  cudaFree(d_jcount);
  cudaFree(d_jmapinv);
  cudaFree(d_fdelta);
  cudaFree(d_tdeltaj);
#endif
#ifdef AOS_LAYOUT
  cudaFree(d_particle_list);
#else
  cudaFree(d_particle_list.idx);
  cudaFree(d_particle_list.x);
  cudaFree(d_particle_list.v);
  cudaFree(d_particle_list.omega);
  cudaFree(d_particle_list.radius);
  cudaFree(d_particle_list.mass);
  cudaFree(d_particle_list.type);
#endif
  cudaFree(d_numneigh);
#ifdef AOS_LAYOUT
  cudaFree(d_neigh);
#else
  cudaFree(d_neigh.idx);
  cudaFree(d_neigh.x);
  cudaFree(d_neigh.v);
  cudaFree(d_neigh.omega);
  cudaFree(d_neigh.radius);
  cudaFree(d_neigh.mass);
  cudaFree(d_neigh.type);
#endif
  cudaFree(d_shear);
  cudaFree(d_force);
  cudaFree(d_torque);
  cudaFree(d_fake_x);
  cudaFree(d_fake_v);
  cudaFree(d_fake_omega);
}
