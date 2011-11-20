#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#ifdef GPU_TIMER
  #include "cuda_timer.h"
#else
  #include "simple_timer.h"
#endif

#include "unpickle.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sched.h>
#include <sstream>
#include <string>
#include <vector>

std::vector<SimpleTimer> one_time;
std::vector<SimpleTimer> per_iter;
std::vector<std::vector<double> > per_iter_timings;

int rand_int(int n);
void shuffle_edges(int *edges, int nedge);
void print_usage(std::string progname);
double percentage_error(double const&expected, double const &computed);
double compare(const char *tag,
    double const&expected, double const&computed, const double threshold,
    bool verbose, bool die_on_flag);
void check_result(struct params *p, NeighListLike *nl,
    double *force, double *torque, double **shearlist,
    const double threshold=0.5, bool verbose=false, bool die_on_flag=true);

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
  printf("   -x arg     set cl platform  ]            \n");
  printf("   -y arg     set cl device    ] OpenCL only\n");
  printf("   -z arg     set cl flags     ]            \n");
}

int main(int argc, char **argv) {
  // only run on main CPU
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(0, &mask);
  int err = sched_setaffinity(0, sizeof(mask), &mask);
  assert(err == 0);

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
  int num_iter = 1000;
  std::string part_filename;
  long seed = -1;
  p->verbose = false;
  p->cl_platform = 0;
  p->cl_device = 0;

  int c;
  while ((c = getopt (argc, argv, "hdvn:p:s:x:y:z:")) != -1) {
    switch (c) {
      case 'h':
        print_usage(progname);
        return 1;
      case 'd':
        debug = true;
        break;
      case 'v':
        p->verbose = true;
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
      case 'x':
        p->cl_platform = atoi(optarg);
        break;
      case 'y':
        p->cl_device = atoi(optarg);
        break;
      case 'z':
        p->cl_flags = optarg;
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
    const char *sname = step_filename.c_str();
    const char *pname = part_filename.c_str();
    printf ("# Command-line parsing: step_filename=%s verbose=%d num_iter=%d part_filename=%s seed=%ld cl_platform=%d cl_device=%d cl_flags=[%s]\n",
        sname, p->verbose, num_iter, pname, seed,
        p->cl_platform, p->cl_device, p->cl_flags);
    for (int i=optind; i<argc; i++)
      printf ("# Non-option argument: %s\n", argv[i]);
  }

  if (p->verbose) {
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

  // PRINT TIMING RESULTS
  assert(per_iter.size() == per_iter_timings.size());
  for (int i=0; i<(int)per_iter_timings.size(); i++) {
    assert(per_iter_timings[i].size() == num_iter);
  }
  double one_time_total = 0.0f;
  double per_iter_total = 0.0f;
  for (int i=0; i<(int)one_time.size(); i++) {
    one_time_total += one_time[i].total_time();
  }
  for (int i=0; i<(int)per_iter.size(); i++) {
    per_iter_total += per_iter[i].total_time();
  }

  if (p->verbose) { //then print header
    printf("# nedge, total_one_time_cost (milliseconds), time_per_iteration");
    for (int i=0; i<(int)one_time.size(); i++) {
      printf(", [%s]", one_time[i].get_name().c_str());
    }
    for (int i=0; i<(int)per_iter.size(); i++) {
      printf(", %s", per_iter[i].get_name().c_str());
    }
    if (seed != -1) {
      printf(", seed");
    }
    printf("\n");
  }

  // print runtime data
  printf("%d, %f, %f", p->nedge, one_time_total, per_iter_total / (double) num_iter);
  for (int i=0; i<(int)one_time.size(); i++) {
    printf(", %f", one_time[i].total_time());
  }
  for (int i=0; i<(int)per_iter.size(); i++) {
    double min = *min_element(per_iter_timings[i].begin(), per_iter_timings[i].end());
    double max = *max_element(per_iter_timings[i].begin(), per_iter_timings[i].end());
    printf(", %f (min %f)(max %f)", per_iter[i].total_time() / (double) num_iter, min, max);
  }
  if (seed != -1) {
    printf(", %ld", seed);
  }
  printf("\n");

  // print out raw sample data
  if (debug) {
    printf("# run");
    for (int i=0; i<(int)per_iter.size(); i++) {
      printf(", %s", per_iter[i].get_name().c_str());
    }
    printf("\n");
    for (int run=0; run<num_iter; run++) {
      printf("%d", run);
      for (int i=0; i<(int)per_iter_timings.size(); i++) {
        printf(", %f", per_iter_timings[i][run]);
      }
      printf("\n");
    }
  }

  delete_params(p);
  return 0;

}

double percentage_error(double const&expected, double const &computed) {
  if (expected == computed) return 0.0;
  // avoid div0 case (send back sentinel value instead)
  // we know computed is not 0.0 because of case above
  if (expected == 0.0) return 999.9;
  return 100 * fabs((computed - expected) / expected);
}

double compare(const char *tag, double const&expected, double const&computed,
               const double threshold,
               bool verbose, bool die_on_flag) {
  static int num_bad = 0;

  double error = percentage_error(expected, computed);
  bool flag = (error > threshold);
  if (flag) {
    num_bad++;
  }
  if (flag || verbose) {
    printf("%s\t%e\t%e\t%e\t%d\n", tag, expected, computed, error, num_bad);
  }
  if (flag && die_on_flag) {
    exit(1);
  }
  return error;
}

void check_result(struct params *p, NeighListLike *nl,
                  double *force, double *torque, double **shearlist,
                  const double threshold, bool verbose, bool die_on_flag) {
  for (int i=0; i<p->nnode*3; i++) {
    std::stringstream tag;
    tag << "force[" << i << "]";
    compare(tag.str().c_str(),
            p->expected_force[i], force[i],
            threshold, verbose, die_on_flag);
  }
  for (int i=0; i<p->nnode*3; i++) {
    std::stringstream tag;
    tag << "torque[" << i << "]";
    compare(tag.str().c_str(),
            p->expected_torque[i], torque[i],
            threshold, verbose, die_on_flag);
  }
  int ptr = 0;
  double *shear_check = new double[p->nedge*3];
  for (int ii=0; ii<nl->inum; ii++) {
    int i = nl->ilist[ii];
    for (int jj=0; jj<nl->numneigh[i]; jj++) {
      double *shear = &(shearlist[i][3*jj]);
      shear_check[(ptr*3)  ] = shear[0];
      shear_check[(ptr*3)+1] = shear[1];
      shear_check[(ptr*3)+2] = shear[2];
      ptr++;
    }
  }
  assert(ptr == p->nedge);
  for (int i=0; i<p->nedge; i++) {
    std::stringstream tag;
    tag << "shear[" << i << "]";
    compare(tag.str().c_str(),
            p->expected_shear[i], shear_check[i],
            threshold, verbose, die_on_flag);
  }
  delete[] shear_check;
}
#endif
