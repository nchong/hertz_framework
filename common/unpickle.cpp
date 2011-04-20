#include "unpickle.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

void print_params(struct params *p) {
  cout << "CONSTANTS" << endl;
  cout << "dt = " << p->dt << endl;
  cout << "nktv2p = " << p->nktv2p << endl;
  cout << "ntype = " << p->ntype << endl;
  for (int i=0; i<p->ntype*p->ntype; i++) {
    cout << "yeff[" << i << "] = " << p->yeff[i] << endl;
  }
  for (int i=0; i<p->ntype*p->ntype; i++) {
    cout << "geff[" << i << "] = " << p->geff[i] << endl;
  }
  for (int i=0; i<p->ntype*p->ntype; i++) {
    cout << "betaeff[" << i << "] = " << p->betaeff[i] << endl;
  }
  for (int i=0; i<p->ntype*p->ntype; i++) {
    cout << "coeffFrict[" << i << "] = " << p->coeffFrict[i] << endl;
  }

  cout << "NODES" << endl;
  cout << "nnode = " << p->nnode << endl;

  cout << "EDGES" << endl;
  cout << "nedge = " << p->nedge << endl;
}

//unpickle array
template<class T>
inline void fill_array(std::ifstream &file, T *array, int num_elements) {
  if (file.eof()) {
    cout << "Error unexpected eof!" << endl;
    exit(-1);
  }
  for (int i=0; i<num_elements; i++) {
    file >> array[i];
  }
}

//unpickle file
struct params *parse_file(std::string fname) {
  ifstream file (fname.c_str(), ifstream::in);
  if (!file.is_open()) {
    cout << "Could not open [" << fname << "]" << endl;
    exit(-1);
  }
  if (file.bad()) {
    cout << "Error with file [" << fname << "]" << endl;
    exit(-1);
  }

  struct params *result = new params;
  int ntype;
  int nnode;
  int nedge;

  //constants
  file >> result->dt;
  file >> result->nktv2p;
  file >> ntype; result->ntype = ntype;
  result->yeff       = new double[ntype*ntype];
  result->geff       = new double[ntype*ntype];
  result->betaeff    = new double[ntype*ntype];
  result->coeffFrict = new double[ntype*ntype];
  fill_array(file, result->yeff,       (ntype*ntype));
  fill_array(file, result->geff,       (ntype*ntype));
  fill_array(file, result->betaeff,    (ntype*ntype));
  fill_array(file, result->coeffFrict, (ntype*ntype));

  //node data
  file >> nnode; result->nnode = nnode;
  result->x      = new double[nnode*3];
  result->v      = new double[nnode*3];
  result->omega  = new double[nnode*3];
  result->radius = new double[nnode  ];
  result->mass   = new double[nnode  ];
  result->type   = new int[nnode];
  result->force  = new double[nnode*3];
  result->torque = new double[nnode*3];
  fill_array(file, result->x,      nnode*3);
  fill_array(file, result->v,      nnode*3);
  fill_array(file, result->omega,  nnode*3);
  fill_array(file, result->radius, nnode);
  fill_array(file, result->mass,   nnode);
  fill_array(file, result->type,   nnode);
  fill_array(file, result->force,  nnode*3);
  fill_array(file, result->torque, nnode*3);

  //edge data
  file >> nedge; result->nedge = nedge;
  result->edge = new int[nedge*2];
  result->shear = new double[nedge*3];
  fill_array(file, result->edge,  nedge*2);
  fill_array(file, result->shear, nedge*3);

  //expected results
  result->expected_force  = new double[nnode*3];
  result->expected_torque = new double[nnode*3];
  result->expected_shear = new double[nedge*3];
  fill_array(file, result->expected_force,  nnode*3);
  fill_array(file, result->expected_torque, nnode*3);
  fill_array(file, result->expected_shear, nedge*3);

  return result;
}
