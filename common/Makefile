include ../Makefile.common

MODULES = unpickle.o inverse_map.o neigh_list.o

all: libcommon.so

libcommon.so: $(MODULES)
	$(CXX) $(CXXFLAGS) $(SHARED) $^ -o $@

clean:
	rm -f $(MODULES) libcommon.so
