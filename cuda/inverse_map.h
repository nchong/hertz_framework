#ifndef INVERSE_MAP_H
#define INVERSE_MAP_H

//#define MAP_BUILD_CHECK //< bounds and sanity checking in build_inverse_map

#ifdef MAP_BUILD_CHECK
  #include "assert.h"
  #warning "Inverse map building will be slower due to bounds/sanity checks."
#endif

// serial implementation of exclusive scan
int *exclusive_scan(int *array, int n);

/*
 * Given a map[N] which implicitly defines a mapping from the set 
 * {0..(N-1)} to {0..(K-1)} 
 * we return a triplet representing the inverse mapping:
 *    - offset[K]
 *    - count[K]
 *    - imap[N]
 *
 * For all dest in {0..K}, offset[dest] and count[dest] give a range of 
 * indexes into imap that contains the src elements that point to 
 * dest in the given map.
 *
 */
void build_inverse_map(
  int *map, int N, int K,                 //inputs
  int *&offset, int *&count, int*&imap);  //outputs

#endif
