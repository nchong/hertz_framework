#include "unpickle.h"
#include "framework.h"
#include "decode.cu"

void run(struct params *input, int num_iter) {
  NeighListLike *nl = new NeighListLike(input);

  one_time.push_back(SimpleTimer("new_gpuneighlist"));
  one_time.back().start();
  GpuNeighList *gnl = new GpuNeighList(nl->inum, nl->maxpage, nl->pgsize);
  one_time.back().stop_and_add_to_total();

  one_time.push_back(SimpleTimer("reload_gpuneighlist"));
  one_time.back().start();
  gnl->reload(nl->numneigh, nl->firstneigh, nl->pages);
  printf("fill-ratio %f\n", gnl->fill_ratio());
  one_time.back().stop_and_add_to_total();
}
