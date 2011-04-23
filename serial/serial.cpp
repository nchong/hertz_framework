/*
 * Serial implementation of the hertz pairwise kernel.
 *
 */

#ifdef GPU_TIMER
  #include "cuda_timer.h"
#else
  #include "simple_timer.h"
#endif

#include "check_result_vector.h"
#include "framework.h"
#include "hertz_constants.h"
#include <math.h>
#include <sstream>

using namespace std;

void run(struct params *input, int num_iter) {

  //--------------------
  // Per-iteration costs
  //--------------------

  per_iter.push_back(SimpleTimer("kernel"));

  for (int run=0; run<num_iter; run++) {

    //make some internal copies
    double *force = new double[input->nnode*3];
    double *torque = new double[input->nnode*3];
    double *shear = new double[input->nedge*3];

    for (int n=0; n<input->nnode*3; n++) {
      force[n] = input->force[n];
      torque[n] = input->torque[n];
    }
    for (int e=0; e<input->nedge*3; e++) {
      shear[e] = input->shear[e];
    }

    //TODO: don't hardcode, push these into constant memory
    double dt = 0.00001;
    double nktv2p = 1;
    double yeff = 3134796.2382445144467056;
    double geff = 556173.5261401557363570;
    double betaeff = -0.3578571305033167;
    double coeffFrict = 0.5;

    per_iter[0].start();
    for (int e=0; e<input->nedge; e++) {
      int i = input->edge[(e*2)];
      int j = input->edge[(e*2)+1];

      double delx = input->x[(i*3)  ] - input->x[(j*3)  ];
      double dely = input->x[(i*3)+1] - input->x[(j*3)+1];
      double delz = input->x[(i*3)+2] - input->x[(j*3)+2];

      double rsq = delx*delx + dely*dely + delz*delz;
      double radsum = input->radius[i] + input->radius[j];
      if (rsq >= radsum*radsum) {
        //unset non-touching atoms
        shear[(e*3)  ] = 0.0;
        shear[(e*3)+1] = 0.0;
        shear[(e*3)+2] = 0.0;
      } else {
        //distance between centres of atoms i and j
        //or, magnitude of del vector
        double r = sqrt(rsq);
        double rinv = 1.0/r;
        double rsqinv = 1.0/rsq;

        // relative translational velocity
        double vr1 = input->v[(i*3)  ] - input->v[(j*3)  ];
        double vr2 = input->v[(i*3)+1] - input->v[(j*3)+1];
        double vr3 = input->v[(i*3)+2] - input->v[(j*3)+2];

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
        double wr1 = (input->radius[i]*input->omega[(i*3)  ] + 
                      input->radius[j]*input->omega[(j*3)  ]) * rinv;
        double wr2 = (input->radius[i]*input->omega[(i*3)+1] + 
                      input->radius[j]*input->omega[(j*3)+1]) * rinv;
        double wr3 = (input->radius[i]*input->omega[(i*3)+2] + 
                      input->radius[j]*input->omega[(j*3)+2]) * rinv;

        // normal forces = Hookian contact + normal velocity damping
        double meff = input->mass[i]*input->mass[j]/(input->mass[i]+input->mass[j]);
        //not-implemented: freeze_group_bit

        double deltan = radsum-r;

        //derive contact model parameters (inlined)
        //yeff, geff, betaeff, coeffFrict are constant lookup tables
        double reff = input->radius[i] * input->radius[j] / (input->radius[i] + input->radius[j]);
        double sqrtval = sqrt(reff * deltan);
        double Sn = 2.    * yeff * sqrtval;
        double St = 8.    * geff * sqrtval;
        double kn = 4./3. * yeff * sqrtval;
        double kt = St;
        double gamman=-2.*sqrtFiveOverSix*betaeff*sqrt(Sn*meff);
        double gammat=-2.*sqrtFiveOverSix*betaeff*sqrt(St*meff);
        double xmu=coeffFrict;
        //not-implemented if (dampflag == 0) gammat = 0;
        kn /= nktv2p;
        kt /= nktv2p;

        double damp = gamman*vnnr*rsqinv;
        double ccel = kn*(radsum-r)*rinv - damp;

        //not-implemented cohesionflag

        // relative velocities
        double vtr1 = vt1 - (delz*wr2-dely*wr3);
        double vtr2 = vt2 - (delx*wr3-delz*wr1);
        double vtr3 = vt3 - (dely*wr1-delx*wr2);

        // shear history effects
        shear[(e*3)  ] += vtr1 * dt;
        shear[(e*3)+1] += vtr2 * dt;
        shear[(e*3)+2] += vtr3 * dt;

        // rotate shear displacements
        double rsht = shear[(e*3)  ]*delx + 
          shear[(e*3)+1]*dely + 
          shear[(e*3)+2]*delz;
        rsht *= rsqinv;

        shear[(e*3)  ] -= rsht*delx;
        shear[(e*3)+1] -= rsht*dely;
        shear[(e*3)+2] -= rsht*delz;

        // tangential forces = shear + tangential velocity damping
        double fs1 = - (kt*shear[(e*3)  ] + gammat*vtr1);
        double fs2 = - (kt*shear[(e*3)+1] + gammat*vtr2);
        double fs3 = - (kt*shear[(e*3)+2] + gammat*vtr3);

        // rescale frictional displacements and forces if needed
        double fs = sqrt(fs1*fs1 + fs2*fs2 + fs3*fs3);
        double fn = xmu * fabs(ccel*r);
        double shrmag = 0;
        if (fs > fn) {
          shrmag = sqrt(
              shear[(e*3)  ]*shear[(e*3)  ] +
              shear[(e*3)+1]*shear[(e*3)+1] +
              shear[(e*3)+2]*shear[(e*3)+2]);
          if (shrmag != 0.0) {
            shear[(e*3)  ] = (fn/fs) * (shear[(e*3)  ] + gammat*vtr1/kt) - gammat*vtr1/kt;
            shear[(e*3)+1] = (fn/fs) * (shear[(e*3)+1] + gammat*vtr2/kt) - gammat*vtr2/kt;
            shear[(e*3)+2] = (fn/fs) * (shear[(e*3)+2] + gammat*vtr3/kt) - gammat*vtr3/kt;
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
        force[(i*3)  ] += fx;
        force[(i*3)+1] += fy;
        force[(i*3)+2] += fz;

        force[(j*3)  ] -= fx;
        force[(j*3)+1] -= fy;
        force[(j*3)+2] -= fz;

        torque[(i*3)  ] -= input->radius[i]*tor1;
        torque[(i*3)+1] -= input->radius[i]*tor2;
        torque[(i*3)+2] -= input->radius[i]*tor3;

        torque[(j*3)  ] -= input->radius[j]*tor1;
        torque[(j*3)+1] -= input->radius[j]*tor2;
        torque[(j*3)+2] -= input->radius[j]*tor3;
      }
    }
    per_iter[0].stop_and_add_to_total();

    //only check results the first time around
    if (run == 0) {
      for (int n=0; n<input->nnode; n++) {
        std::stringstream out;
        out << "force[" << n << "]";
        check_result_vector(
            out.str().c_str(),
            &input->expected_force[(n*3)], &force[(n*3)]);
        out.str("");

        out << "torque[" << n << "]";
        check_result_vector(
            out.str().c_str(),
            &input->expected_torque[(n*3)], &torque[(n*3)]);
      }
      for (int n=0; n<input->nedge; n++) {
        stringstream out;
        out << "shear[" << n << "]";
        check_result_vector(
            out.str().c_str(),
            &input->expected_shear[(n*3)], &shear[(n*3)]);
      }
    }

  }
}
