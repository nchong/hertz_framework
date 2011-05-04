#ifndef SIMPLE_TIMER_H
#define SIMPLE_TIMER_H

#include <sys/time.h>
#include <errno.h>
#include <unistd.h>
#include <string>
#include <assert.h>
#include <cstdio>

/*
 * A stopwatch that counts in milliseconds.
 *
 *     SimpleTimer t("Kernel");
 *     for (...) {
 *       t.start();
 *       //kernel
 *       t.stop_and_add_to_total();
 *     }
 *     time_in_ms = t.total_time();
 *
 */
class SimpleTimer {
  public:
    SimpleTimer() : _total_time(0.0f) {}
    SimpleTimer(std::string name) : _total_time(0.0f), _name(name) {}
    ~SimpleTimer() {}

    inline void start() {
      int err = clock_gettime(CLOCK_REALTIME, &_start);
      assert (err == 0);
      assert (_start.tv_nsec > 0);
      assert (_start.tv_nsec < 1000000000);
    };
    inline void stop()  {
      int err = clock_gettime(CLOCK_REALTIME, &_end);
      assert (err == 0);
      assert (_end.tv_nsec > 0);
      assert (_end.tv_nsec < 1000000000);
    };
    inline void add_to_total() {
      _total_time +=
        (double)(_end.tv_sec  - _start.tv_sec )*1.0e3 +
        (double)(_end.tv_nsec - _start.tv_nsec)/1.0e6;
      //long seconds = _end.tv_sec - _start.tv_sec;
      //long nseconds = _end.tv_nsec - _start.tv_nsec;
      //_total_time += ((seconds*1000) + (nseconds/1000000.0));
    };
    inline void stop_and_add_to_total() {
      stop();
      add_to_total();
    };
    inline double total_time() { return _total_time; };
    inline void reset() { _total_time = 0.0; };
    inline std::string get_name() { return _name; };
    inline void set_name(const char *s) { _name.assign(s); };
    inline void set_total_time(double t) { _total_time = t; };

  private:
    std::string _name;
    struct timespec _start, _end;
    double _total_time;
};

#endif
