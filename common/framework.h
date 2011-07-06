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

/* randomly shuffle the contact list (array of pairs) */
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

void print_usage(std::string progname) {
  printf("Usage: %s <stepfile> [options]\n", progname.c_str());
  printf("Options:\n");
  printf("   -n arg     number of runs\n");
  printf("   -v         be verbose\n");
  printf("   -p arg     use partition file\n");
  printf("   -s arg     set seed for edge shuffle\n");
}

int main(int argc, char **argv) {

  // PARSE CMDLINE
  std::string progname(argv[0]);

  // mandatory arguments
  if (argc < 2) {
    print_usage(progname);
    return(1);
  }
  std::string step_filename(argv[1]);
  struct params *p = parse_file(step_filename);
  argc--;
  argv++;

  // optional arguments
  bool debug = false;
  bool verbose = false;
  int num_iter = 1000;
  std::string part_filename;
  long seed = -1;

  int c;
  while ((c = getopt (argc, argv, "hdvn:p:s:")) != -1) {
    switch (c) {
      case 'h':
        print_usage(progname);
        return 1;
      case 'd':
        debug = true;
        break;
      case 'v':
        verbose = true;
        break;
      case 'n':
        num_iter = atoi(optarg);
        break;
      case 'p':
        part_filename = optarg;
        parse_partition_file(p, part_filename);
        break;
      case 's':
        seed = atol(optarg);
        srandom(seed);
        shuffle_edges(p->edge, p->nedge);
        break;
      case '?':
        if (optopt == 'n' || optopt == 'p' || optopt == 's')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr,
              "Unknown option character `\\x%x'.\n",
              optopt);
        return 1;
      default:
        abort ();
    }
  }

  if (debug) {
    printf ("# Command-line parsing: step_filename=%s verbose=%d num_iter=%d part_filename=%s seed=%ld\n",
        step_filename.c_str(), verbose, num_iter, part_filename.c_str(), seed);
    for (int i=optind; i<argc; i++)
      printf ("# Non-option argument: %s\n", argv[i]);
  }

  if (verbose) {
    printf("# Program: %s\n", progname.c_str());
    printf("# Num Iterations: %d\n", num_iter);
    if (p->npartition > 1) {
      printf("# Partition: %s\n", part_filename.c_str());
      printf("# npartition: %d\n", p->npartition);
    }
    if (seed != -1) {
      printf("# Shuffle seed: %ld\n", seed);
    }
#ifdef GPU_TIMER
    printf("# GPU timer implementation\n");
#elif POSIX_TIMER
    printf("# POSIX timer implementation\n");
#else
    printf("# CPU timer implementation\n");
#endif
  }

  // RUN TEST
  run(p, num_iter);
  double one_time_total = 0.0f;
  double per_iter_total = 0.0f;
  for (int i=0; i<one_time.size(); i++) {
    one_time_total += one_time[i].total_time();
  }
  for (int i=0; i<per_iter.size(); i++) {
    per_iter_total += per_iter[i].total_time();
  }

  if (verbose) { //then print header
    printf("# nedge, total_one_time_cost (milliseconds), time_per_iteration");
    for (int i=0; i<one_time.size(); i++) {
      printf(", [%s]", one_time[i].get_name().c_str());
    }
    for (int i=0; i<per_iter.size(); i++) {
      printf(", %s", per_iter[i].get_name().c_str());
    }
    if (seed != -1) {
      printf(", seed");
    }
    printf("\n");
  }

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

  delete_params(p);
  return 0;

}
#endif
