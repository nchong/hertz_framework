#include "unpickle.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <limits>

using namespace std;

void print_params(struct params *p) {
  cout << "CONSTANTS" << endl;
  cout.precision(numeric_limits<double>::digits10);
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
    file.read(reinterpret_cast<char *>(&(array[i])), sizeof(array[i]));
  }
}

//unpickle file
struct params *parse_file(std::string fname) {
  const char *fname_c_str = fname.c_str();
  ifstream file (fname_c_str, ifstream::in | ifstream::binary);
  if (!file.is_open()) {
    cout << "Could not open [" << fname << "]" << endl;
    exit(-1);
  }
  if (file.bad()) {
    cout << "Error with file [" << fname << "]" << endl;
    exit(-1);
  }

  unsigned int EXPECTED_MAGIC = 0xDEADBEEF;
  unsigned int MAGIC;
  file.read(reinterpret_cast<char *>(&MAGIC), sizeof(MAGIC));
  if (MAGIC != 0xDEADBEEF) {
    printf("Error with file [%s]: magic number is [%x]; expecting [%x]\n", fname.c_str(), MAGIC, EXPECTED_MAGIC);
    exit(-1);
  }

  struct params *result = new params;
  int ntype;
  int nnode;
  int nedge;

  //constants
  file.read(reinterpret_cast<char *>(&(result->dt)), sizeof(result->dt));
  file.read(reinterpret_cast<char *>(&(result->nktv2p)), sizeof(result->nktv2p));
  file.read(reinterpret_cast<char *>(&(ntype)), sizeof(ntype)); result->ntype = ntype;
  result->yeff       = new double[ntype*ntype];
  result->geff       = new double[ntype*ntype];
  result->betaeff    = new double[ntype*ntype];
  result->coeffFrict = new double[ntype*ntype];
  fill_array(file, result->yeff,       (ntype*ntype));
  fill_array(file, result->geff,       (ntype*ntype));
  fill_array(file, result->betaeff,    (ntype*ntype));
  fill_array(file, result->coeffFrict, (ntype*ntype));

  //node data
  file.read(reinterpret_cast<char *>(&(nnode)), sizeof(nnode)); result->nnode = nnode;
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
  file.read(reinterpret_cast<char *>(&(nedge)), sizeof(nedge)); result->nedge = nedge;
  result->edge = new int[nedge*2];
  result->shear = new double[nedge*3];
  fill_array(file, result->edge,  nedge*2);
  fill_array(file, result->shear, nedge*3);

  //partition data (OP2 only)
  //by default no partition data means we just use the unpickled edge ordering
  result->npartition = 1;
  result->partition_length = vector<int>(1, nedge);

  //expected results
  result->expected_force  = new double[nnode*3];
  result->expected_torque = new double[nnode*3];
  result->expected_shear = new double[nedge*3];
  fill_array(file, result->expected_force,  nnode*3);
  fill_array(file, result->expected_torque, nnode*3);
  fill_array(file, result->expected_shear, nedge*3);

  return result;
}

void parse_partition_file(struct params *input, string fname) {
  ifstream file (fname.c_str(), ifstream::in);
  if (!file.is_open()) {
    cout << "Could not open [" << fname << "]" << endl;
    exit(-1);
  }
  if (file.bad()) {
    cout << "Error with file [" << fname << "]" << endl;
    exit(-1);
  }

  //number of partitions
  int npartition;
  file >> npartition;
  input->npartition = npartition;

  //node_partition_map is a mapping from nodeid to partitionid
  int *node_partition_map = new int[input->nnode];
  fill_array(file, node_partition_map, input->nnode);

  //edge_partition_map is a mapping from edgeid to partitionid
  //we use the first node of an edge (part1) to determine an edge's partitionid
  int *edge_partition_map = new int[input->nedge];
  for (int i=0; i<input->nedge; i++) {
    int part1 = node_partition_map[input->edge[(i*2)]];
  //int part2 = node_partition_map[input->edge[(i*2)+1]];
    edge_partition_map[i] = part1;
  }

  //create a new edge set
  int *edge2 = new int[input->nedge*2];
  int *partition_length = new int[npartition];
  int ptr = 0;
  for (int p=0; p<npartition; p++) {
    int nedge_in_p = 0;
    for (int e=0; e<input->nedge; e++) {
      if (edge_partition_map[e] == p) {
        edge2[(ptr*2)  ] = input->edge[(e*2)  ];
        edge2[(ptr*2)+1] = input->edge[(e*2)+1];
        ptr++;
        nedge_in_p++;
      }
    }
    partition_length[p] = nedge_in_p;
  }

  //copy new edge set over original input
	memcpy(input->edge, edge2, sizeof(int)*input->nedge*2);

  //create a new shear set
  double *shear2 = new double[input->nedge*3];
  double *expected_shear2 = new double[input->nedge*3];
  ptr = 0;
  for (int p=0; p<npartition; p++) {
    for (int e=0; e<input->nedge; e++) {
      if (edge_partition_map[e] == p) {
        shear2[(ptr*3)  ] = input->shear[(e*3)  ];
        shear2[(ptr*3)+1] = input->shear[(e*3)+1];
        shear2[(ptr*3)+2] = input->shear[(e*3)+2];

        expected_shear2[(ptr*3)  ] = input->expected_shear[(e*3)  ];
        expected_shear2[(ptr*3)+1] = input->expected_shear[(e*3)+1];
        expected_shear2[(ptr*3)+2] = input->expected_shear[(e*3)+2];

        ptr++;
      }
    }
  }

  //copy new shear set over original input
	memcpy(input->shear, shear2, sizeof(double)*input->nedge*3);
	memcpy(input->expected_shear, expected_shear2, sizeof(double)*input->nedge*3);

  //partition_length is a mapping from partitionid to length (the number of
  //*edges* in the partition)
  input->partition_length =
    vector<int>(partition_length, partition_length + npartition);

  file.close();
}

void delete_params(struct params *p) {
  delete[] p->yeff;
  delete[] p->geff;
  delete[] p->betaeff;
  delete[] p->coeffFrict;

  delete[] p->x;
  delete[] p->v;
  delete[] p->omega;
  delete[] p->radius;
  delete[] p->mass;
  delete[] p->type;
  delete[] p->force;
  delete[] p->torque;

  delete[] p->edge;
  delete[] p->shear;

  delete[] p->expected_force;
  delete[] p->expected_torque;
  delete[] p->expected_shear;

  if (p->npartition > 1) {
    p->partition_length.clear();
  }

  delete p;
}

/*
 * Artificially inflate the size of a parameter list
 */
void inflate(struct params *p, int k) {
  // multiply out node data
  int nnode = p->nnode*k;
  double *x               = new double[nnode*3];
  double *v               = new double[nnode*3];
  double *omega           = new double[nnode*3];
  double *radius          = new double[nnode];
  double *mass            = new double[nnode];
  int    *type            = new int[nnode];
  double *force           = new double[nnode*3];
  double *torque          = new double[nnode*3];
  double *expected_force  = new double[nnode*3];
  double *expected_torque = new double[nnode*3];
  for (int i=0; i<k; i++) {
    copy(p->x,               p->x              +(p->nnode*3), &x[i*p->nnode*3]);
    copy(p->v,               p->v              +(p->nnode*3), &v[i*p->nnode*3]);
    copy(p->omega,           p->omega          +(p->nnode*3), &omega[i*p->nnode*3]);
    copy(p->radius,          p->radius         +(p->nnode),   &radius[i*p->nnode]);
    copy(p->mass,            p->mass           +(p->nnode),   &mass[i*p->nnode]);
    copy(p->type,            p->type           +(p->nnode),   &type[i*p->nnode]);
    copy(p->force,           p->force          +(p->nnode*3), &force[i*p->nnode*3]);
    copy(p->torque,          p->torque         +(p->nnode*3), &torque[i*p->nnode*3]);
    copy(p->expected_force,  p->expected_force +(p->nnode*3), &expected_force[i*p->nnode*3]);
    copy(p->expected_torque, p->expected_torque+(p->nnode*3), &expected_torque[i*p->nnode*3]);
  }

  // multiply out edge data
  int nedge = p->nedge*k;
  int    *edge           = new int[nedge*2];
  double *shear          = new double[nedge*3];
  double *expected_shear = new double[nedge*3];
  for (int i=0; i<k; i++) {
    copy(p->edge,           p->edge          +(p->nedge*2), &edge[i*p->nedge*2]);
    copy(p->shear,          p->shear         +(p->nedge*3), &shear[i*p->nedge*3]);
    copy(p->expected_shear, p->expected_shear+(p->nedge*3), &expected_shear[i*p->nedge*3]);
  }
  for (int i=1; i<k; i++) {
    for (int e=0; e<p->nedge*2; e++) {
      edge[(p->nedge*2*i)+e] += (p->nnode*i);
    }
  }
#if 0 //paranoid
  for (int i=0; i<p->nnode*3; i++) {
    for (int j=1; j<k; j++) {
      assert(x[i] == x[i+(j*p->nnode*3)]);
    }
  }
  for (int i=0; i<p->nedge*2; i++) {
    for (int j=1; j<k; j++) {
      assert(edge[i] + (j*p->nnode) == edge[i+(j*p->nedge*2)]);
    }
  }
#endif
  delete[] p->x;
  delete[] p->v;
  delete[] p->omega;
  delete[] p->radius;
  delete[] p->mass;
  delete[] p->type;
  delete[] p->force;
  delete[] p->torque;

  delete[] p->edge;
  delete[] p->shear;

  delete[] p->expected_force;
  delete[] p->expected_torque;
  delete[] p->expected_shear;

  p->nnode  = nnode;
  p->x      = x;
  p->v      = v;
  p->omega  = omega;
  p->radius = radius;
  p->mass   = mass;
  p->type   = type;
  p->force  = force;
  p->torque = torque;

  p->nedge = nedge;
  p->edge  = edge;
  p->shear = shear;

  p->expected_force  =  expected_force;
  p->expected_torque = expected_torque;
  p->expected_shear  = expected_shear;
}
