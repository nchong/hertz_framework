#ifndef SIMPLE_TIMER_H
#define SIMPLE_TIMER_H

#include <sys/time.h>
#include <unistd.h>
#include <string>

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

    inline void start() { gettimeofday(&_start, NULL); };
    inline void stop() { gettimeofday(&_end, NULL); };
    inline double add_to_total() { 
      long seconds = _end.tv_sec - _start.tv_sec;
      long useconds = _end.tv_usec - _start.tv_usec;
      double delta = ((seconds) * 1000 + useconds/1000.0) + 0.5;
      _total_time += delta;
      return (double)delta;
    };
    inline double stop_and_add_to_total() {
      stop();
      return add_to_total();
    };
    inline double total_time() { return _total_time; };
    inline void reset() { _total_time = 0.0; };
    inline std::string get_name() { return _name; };
    inline void set_name(const char *s) { _name.assign(s); };
    inline void set_total_time(double t) { _total_time = t; };

  private:
    double _total_time;
    std::string _name;
    struct timeval _start, _end;
};

#endif
