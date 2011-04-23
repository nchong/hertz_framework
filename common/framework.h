#ifdef GPU_TIMER
  #include "cuda_timer.h"
#else
  #include "simple_timer.h"
#endif
#include "unpickle.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

std::vector<SimpleTimer> one_time;
std::vector<SimpleTimer> per_iter;

/*
 * Run [num_iter] iterations of the hertz computation. [one_time] and [per_iter]
 * store timing results for one-time and per-iteration costs, respectively.
 */
extern void run(struct params *input, int num_iter);

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <step_file> [num_iterations] [partition_file]\n", argv[0]);
    return(-1);
  }

  std::string step_filename(argv[1]);
  struct params *p = parse_file(step_filename);

  int num_iter = 1000;
  if (argc > 2) {
    num_iter = atoi(argv[2]);
  }

  std::string partition_filename("");
  if (argc > 3) {
    partition_filename = argv[3];
    parse_partition_file(p, partition_filename);
  }

  printf("# Program: %s\n", argv[0]);
  printf("# Num Iterations: %d\n", num_iter);
  if (p->npartition > 1) {
    printf("# Partition: %s\n", partition_filename.c_str());
    printf("# npartition: %d\n", p->npartition);
  }
#ifdef GPU_TIMER
  printf("# GPU timer implementation\n");
#else
  printf("# CPU timer implementation\n");
#endif

  run(p, num_iter);

  double one_time_total;
  double per_iter_total;
  printf("# nedge, total_one_time_cost (milliseconds), time_per_iteration");
  for (int i=0; i<one_time.size(); i++) {
    printf(", [%s]", one_time[i].get_name().c_str());
    one_time_total += one_time[i].total_time();
  }
  for (int i=0; i<per_iter.size(); i++) {
    printf(", %s", per_iter[i].get_name().c_str());
    per_iter_total += per_iter[i].total_time();
  }
  printf("\n");

  printf("%d, %f, %f", p->nedge, one_time_total, per_iter_total / (double) num_iter);
  for (int i=0; i<one_time.size(); i++) {
    printf(", %f", one_time[i].total_time());
  }
  for (int i=0; i<per_iter.size(); i++) {
    printf(", %f", per_iter[i].total_time() / (double) num_iter);
  }
  printf("\n");

  return(0);
}
