#include "unpickle.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

template <typename T>
void print_array(ofstream &ofile, T *x, int len) {
  for (int i=0; i<len; i++) {
    ofile.write(reinterpret_cast<char *>(&(x[i])), sizeof(x[i]));
  }
}

int main(int argc, char **argv) {
  std::string progname(argv[0]);
  if (argc < 3) {
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
  ofstream ofile;
  ofile.open(argv[2], ios::binary | ios::out);

  unsigned int MAGIC = 0xDEADBEEF;
  ofile.write(reinterpret_cast<char *>(&MAGIC), sizeof(MAGIC));

  //CONSTANTS
  ofile.write(reinterpret_cast<char *>(&(p->dt)), sizeof(p->dt));
  ofile.write(reinterpret_cast<char *>(&(p->nktv2p)), sizeof(p->nktv2p));
  ofile.write(reinterpret_cast<char *>(&(p->ntype)), sizeof(p->ntype));
  print_array<double>(ofile, p->yeff, p->ntype*p->ntype);
  print_array<double>(ofile, p->geff, p->ntype*p->ntype);
  print_array<double>(ofile, p->betaeff, p->ntype*p->ntype);
  print_array<double>(ofile, p->coeffFrict, p->ntype*p->ntype);

  ofile.write(reinterpret_cast<char *>(&(p->nnode)), sizeof(p->nnode));
  print_array<double>(ofile, p->x, p->nnode*3);
  print_array<double>(ofile, p->v, p->nnode*3);
  print_array<double>(ofile, p->omega, p->nnode*3);
  print_array<double>(ofile, p->radius, p->nnode);
  print_array<double>(ofile, p->mass, p->nnode);
  print_array<int>(ofile, p->type, p->nnode);
  print_array<double>(ofile, p->force, p->nnode*3);
  print_array<double>(ofile, p->torque, p->nnode*3);

  int new_nedge = p->nedge*2;
  ofile.write(reinterpret_cast<char *>(&(new_nedge)), sizeof(new_nedge));
  int count = 0;
  for (int n=0; n<p->nnode; n++) {
    for (int i=0; i<adj[n].size(); i++) {
      ofile.write(reinterpret_cast<char *>(&(n)), sizeof(n));
      ofile.write(reinterpret_cast<char *>(&(adj[n][i])), sizeof(adj[n][i]));
      count++;
    }
  }

  double nshear[3];
  assert(count == p->nedge*2);
  for (int n=0; n<p->nnode; n++) {
    for (int i=0; i<adj[n].size(); i++) {
      if (negate[n][i]) {
        nshear[0] = -(p->shear[(back[n][i]*3)+0]);
        nshear[1] = -(p->shear[(back[n][i]*3)+1]);
        nshear[2] = -(p->shear[(back[n][i]*3)+2]);
      } else {
        nshear[0] =  (p->shear[(back[n][i]*3)+0]);
        nshear[1] =  (p->shear[(back[n][i]*3)+1]);
        nshear[2] =  (p->shear[(back[n][i]*3)+2]);
      }
      ofile.write(reinterpret_cast<char *>(&(nshear[0])), sizeof(nshear[0]));
      ofile.write(reinterpret_cast<char *>(&(nshear[1])), sizeof(nshear[1]));
      ofile.write(reinterpret_cast<char *>(&(nshear[2])), sizeof(nshear[2]));
    }
  }

  print_array<double>(ofile, p->expected_force, p->nnode*3);
  print_array<double>(ofile, p->expected_torque, p->nnode*3);
  for (int n=0; n<p->nnode; n++) {
    for (int i=0; i<adj[n].size(); i++) {
      if (negate[n][i]) {
        nshear[0] = -(p->expected_shear[(back[n][i]*3)+0]);
        nshear[1] = -(p->expected_shear[(back[n][i]*3)+1]);
        nshear[2] = -(p->expected_shear[(back[n][i]*3)+2]);
      } else {
        nshear[0] =  (p->expected_shear[(back[n][i]*3)+0]);
        nshear[1] =  (p->expected_shear[(back[n][i]*3)+1]);
        nshear[2] =  (p->expected_shear[(back[n][i]*3)+2]);
      }
      ofile.write(reinterpret_cast<char *>(&(nshear[0])), sizeof(nshear[0]));
      ofile.write(reinterpret_cast<char *>(&(nshear[1])), sizeof(nshear[1]));
      ofile.write(reinterpret_cast<char *>(&(nshear[2])), sizeof(nshear[2]));
    }
  }

  return 0;
}
