#ifndef CHECK_RESULT_VECTOR_H
#define CHECK_RESULT_VECTOR_H

bool check_result_vector(
  const char* id,
  double expected[3], double actual[3], const double epsilon=0.00001,
  bool verbose=false, bool die_on_flag=true);

#endif
