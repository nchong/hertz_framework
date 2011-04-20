#include "inverse_map.h"
#include <assert.h>

int *exclusive_scan(int *array, int n) {
  int *result = new int[n];
  result[0] = 0;
  for (int i=1; i<n; i++) {
    result[i] = result[i-1] + array[i-1];
  }
  return result;
}

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
  int *map, int N, int K,
  int *&offset, int *&count, int*&imap) {
#ifdef MAP_BUILD_CHECK
  assert(offset == NULL);
  assert(count == NULL);
  assert(imap == NULL);
#endif

  //output offset, count and imap
  count = new int[K];
  imap = new int[N];
  for (int i=0; i<K; i++) {
    count[i] = 0;
  }
  //use a temporary (sparse) inverse map so 
  //we only have to scan through map once
  int **sparse_imap = new int*[K];
  for (int i=0; i<K; i++) {
    sparse_imap[i] = new int[32];
  }

  //scan through the map and insert the inverse into sparse map
  for (int src=0; src<N; src++) {
    int dest = map[src];
#ifdef MAP_BUILD_CHECK
    assert(0 <= dest && dest < K);
    assert(count[dest] < 32);
#endif
    sparse_imap[dest][count[dest]] = src;
    count[dest]++;
  }

  //count used to calculate an offset into the packed imap
  offset = exclusive_scan(count, K);

  //squash sparse map into packed imap
  for (int dest=0; dest<K; dest++) {
    for (int i=0; i<count[dest]; i++) {
      assert((offset[dest]+i) < N);
      int src = sparse_imap[dest][i];
      imap[offset[dest]+i] = src;
    }
  }

#ifdef MAP_BUILD_CHECK
  //sanity check that the inverse map is correct
  for (int dest=0; dest<K; dest++) {
    for (int i=0; i<count[dest]; i++) {
      int src = imap[offset[dest]+i];
      if (map[src] != dest) {
        printf("map[%d] = %d != %d\n", src, map[src], dest);
      }
      assert(map[src] == dest);
    }
  }
#endif
}
