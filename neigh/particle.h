#ifndef PARTICLE_H
#define PARTICLE_H

#ifdef AOS_LAYOUT
struct particle {
  int idx;
  double x[3];
  double v[3];
  double omega[3];
  double radius;
  double mass;
  int    type;
};
#else
struct particle {
  int *idx;
  double *x;
  double *v;
  double *omega;
  double *radius;
  double *mass;
  int    *type;
};
#endif

#endif
