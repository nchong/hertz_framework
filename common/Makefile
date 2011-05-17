include ../Makefile.common

MODULES = unpickle.o check_result_vector.o inverse_map.o

libcommon.so: $(MODULES)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@

clean:
	rm -f $(MODULES) libcommon.so
