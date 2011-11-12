#include "unpickle.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

void print_double_array(double *x, int len) {
  for (int i=0; i<len; i++) {
    printf("%.16f\n", x[i]);
  }
}

void print_int_array(int *x, int len) {
  for (int i=0; i<len; i++) {
    printf("%d\n", x[i]);
  }
}

int main(int argc, char **argv) {
  std::string progname(argv[0]);
  if (argc < 2) {
    return 1;
  }
  std::string step_filename(argv[1]);
  struct params *p = parse_file(step_filename);

  std::vector<int> *adj = new std::vector<int>[p->nnode];
  std::vector<int> *back = new std::vector<int>[p->nnode];
  std::vector<bool> *negate = new std::vector<bool>[p->nnode];
  for (int e=0; e<p->nedge; e++) {
   int i = p->edge[(e*2)];
   int j = p->edge[(e*2)+1];
   double *shear = &p->shear[(e*3)];
   double *shear_expected = &p->expected_shear[(e*3)];
   adj[i].push_back(j);
   back[i].push_back(e);
   negate[i].push_back(false);
   adj[j].push_back(i);
   back[j].push_back(e);
   negate[j].push_back(true);
  }

  //repickle
  printf("%.16f\n", p->dt);
  printf("%.16f\n", p->nktv2p);
  printf("%d\n", p->ntype);
  print_double_array(p->yeff, p->ntype*p->ntype);
  print_double_array(p->geff, p->ntype*p->ntype);
  print_double_array(p->betaeff, p->ntype*p->ntype);
  print_double_array(p->coeffFrict, p->ntype*p->ntype);

  printf("%d\n", p->nnode);
  print_double_array(p->x, p->nnode*3);
  print_double_array(p->v, p->nnode*3);
  print_double_array(p->omega, p->nnode*3);
  print_double_array(p->radius, p->nnode);
  print_double_array(p->mass, p->nnode);
  print_int_array(p->type, p->nnode);
  print_double_array(p->force, p->nnode*3);
  print_double_array(p->torque, p->nnode*3);

  printf("%d\n", p->nedge*2);
  int count = 0;
  for (int n=0; n<p->nnode; n++) {
    for (int i=0; i<adj[n].size(); i++) {
      printf("%d\n", n);
      printf("%d\n", adj[n][i]);
      count++;
    }
  }
  assert(count == p->nedge*2);
  for (int n=0; n<p->nnode; n++) {
    for (int i=0; i<adj[n].size(); i++) {
      if (negate[n][i]) {
        printf("%.16f\n", -(p->shear[(back[n][i]*3)+0]));
        printf("%.16f\n", -(p->shear[(back[n][i]*3)+1]));
        printf("%.16f\n", -(p->shear[(back[n][i]*3)+2]));
      } else {
        printf("%.16f\n", p->shear[(back[n][i]*3)+0]);
        printf("%.16f\n", p->shear[(back[n][i]*3)+1]);
        printf("%.16f\n", p->shear[(back[n][i]*3)+2]);
     }
    }
  }

  print_double_array(p->expected_force, p->nnode*3);
  print_double_array(p->expected_torque, p->nnode*3);
  for (int n=0; n<p->nnode; n++) {
    for (int i=0; i<adj[n].size(); i++) {
      if (negate[n][i]) {
        printf("%.16f\n", -(p->expected_shear[(back[n][i]*3)+0]));
        printf("%.16f\n", -(p->expected_shear[(back[n][i]*3)+1]));
        printf("%.16f\n", -(p->expected_shear[(back[n][i]*3)+2]));
      } else {
        printf("%.16f\n", p->expected_shear[(back[n][i]*3)+0]);
        printf("%.16f\n", p->expected_shear[(back[n][i]*3)+1]);
        printf("%.16f\n", p->expected_shear[(back[n][i]*3)+2]);
      }
    }
  }

  return 0;
}
