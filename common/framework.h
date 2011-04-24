#ifndef FRAMEWORK_H
#define FRAMEWORK_H

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

/* return a random integer in [0..n) */
int rand_int(int n) {
  int limit = RAND_MAX - RAND_MAX % n;
  int rnd;

  do {
    rnd = random();
  } while (rnd >= limit);
  return rnd % n;

}

/* randomly shuffle the contact list (an array of pairs) */
void shuffle_edges(int *edges, int nedge) {
  int i, j, n0, n1;

  for (i = nedge - 1; i > 0; i--) {
    j = rand_int(i + 1);
    n0 = edges[(j*2)];
    n1 = edges[(j*2)+1];
    edges[(j*2)]   = edges[(i*2)];
    edges[(j*2)+1] = edges[(i*2)+1];
    edges[(i*2)]   = n0;
    edges[(i*2)+1] = n1;
  }
}

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
    if (partition_filename != "none") {
      parse_partition_file(p, partition_filename);
    }
  }

  long seed = -1;
  if (argc > 4) {
    seed = atol(argv[4]);
    srandom(seed);
    shuffle_edges(p->edge, p->nedge);
  }

  printf("# Program: %s\n", argv[0]);
  printf("# Num Iterations: %d\n", num_iter);
  if (p->npartition > 1) {
    printf("# Partition: %s\n", partition_filename.c_str());
    printf("# npartition: %d\n", p->npartition);
  }
  if (seed != -1) {
    printf("# Shuffle seed: %ld\n", seed);
  }
#ifdef GPU_TIMER
  printf("# GPU timer implementation\n");
#else
  printf("# CPU timer implementation\n");
#endif

  run(p, num_iter);

  // print header
  double one_time_total = 0.0;
  double per_iter_total = 0.0;
  printf("# nedge, total_one_time_cost (milliseconds), time_per_iteration");
  for (int i=0; i<one_time.size(); i++) {
    printf(", [%s]", one_time[i].get_name().c_str());
    one_time_total += one_time[i].total_time();
  }
  for (int i=0; i<per_iter.size(); i++) {
    printf(", %s", per_iter[i].get_name().c_str());
    per_iter_total += per_iter[i].total_time();
  }
  if (seed != -1) {
    printf(", seed");
  }
  printf("\n");

  // print runtime data
  printf("%d, %f, %f", p->nedge, one_time_total, per_iter_total / (double) num_iter);
  for (int i=0; i<one_time.size(); i++) {
    printf(", %f", one_time[i].total_time());
  }
  for (int i=0; i<per_iter.size(); i++) {
    printf(", %f", per_iter[i].total_time() / (double) num_iter);
  }
  if (seed != -1) {
    printf(", %ld", seed);
  }
  printf("\n");

  return(0);
}
#endif
