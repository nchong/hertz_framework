#include "check_result_vector.h"
#include <math.h>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>

//check two vectors against each other (using absolute difference of elements)
bool check_result_vector(
  const char* id, 
  double expected[3], double actual[3], const double epsilon,
  bool verbose, bool die_on_flag) {
  bool flag = (fabs(expected[0] - actual[0]) > epsilon ||
               fabs(expected[1] - actual[1]) > epsilon ||
               fabs(expected[2] - actual[2]) > epsilon);
  const char *marker = flag ? "***" : "   ";

  static int num_bad = 0;
  if (flag) {
    num_bad++;
  }
  if (flag || verbose) {
    std::printf("%s%s: {%.16f, %.16f, %.16f} / {%.16f, %.16f, %.16f}%s(%d)\n",
        marker,
        id,
        expected[0], expected[1], expected[2],
        actual[0], actual[1], actual[2],
        marker, num_bad
        );
  }
  if (flag && die_on_flag) {
    exit(-1);
  }
  return flag;
}
