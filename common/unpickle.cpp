#include "unpickle.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

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
        edge2[ptr*2] = input->edge[e*2];
        edge2[(ptr*2)+1] = input->edge[(e*2)+1];
        ptr++;
        nedge_in_p++;
      }
    }
    partition_length[p] = nedge_in_p;
  }

  //copy new edge set over original input
	memcpy(input->edge, edge2, sizeof(int)*input->nedge*2);

  //partition_length is a mapping from partitionid to length (the number of
  //*edges* in the partition)
  input->partition_length =
    vector<int>(partition_length, partition_length + npartition);
}
