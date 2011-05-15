#ifndef PARTICLE_H
#define PARTICLE_H

struct particle {
  int idx;
  double x[3];
  double v[3];
  double omega[3];
  double radius;
  double mass;
  int    type;
};

#endif
