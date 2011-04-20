#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include "cuda_common.h"
#include <cstdio>
#include <string>

class SimpleTimer {
  public:
    SimpleTimer() : _total_time(0.0f) {
      ASSERT_NO_CUDA_ERROR(cudaEventCreate(&start_event));
      ASSERT_NO_CUDA_ERROR(cudaEventCreate(&stop_event));
    }
    SimpleTimer(std::string name) : _name(name) { SimpleTimer(); }
    ~SimpleTimer() {
      cudaEventDestroy(start_event);
      cudaEventDestroy(stop_event);
    }

    inline void start() {
      ASSERT_NO_CUDA_ERROR(cudaEventRecord(start_event,0));
    }
    inline void stop() {
      ASSERT_NO_CUDA_ERROR(cudaEventRecord(stop_event,0));
    }
    inline void add_to_total() { _total_time+=time(); }
    inline double time() { 
      float timer;
      cudaEventSynchronize(stop_event);
      ASSERT_NO_CUDA_ERROR(
        cudaEventElapsedTime(&timer,start_event,stop_event));
      return timer;
    }
    inline void stop_and_add_to_total() {
      stop();
      add_to_total();
    };
    inline double total_time() { return _total_time; };
    inline void reset() { _total_time = 0.0; };
    inline std::string get_name() { return _name; };
    inline void set_name(const char *s) { _name.assign(s); };

  private:
    std::string _name;
    cudaEvent_t start_event, stop_event;
    double _total_time;
};

#endif
