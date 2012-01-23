include ../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

# ---------------------------------------------------------------------------
# BUILD OPTIONS
# ---------------------------------------------------------------------------
override BLOCK_SIZE ?= 32

# MAX_GRID_DIM should be the maximum number of allocatable threads in the grid's x dimension.
# This is required to determine when to spill into using a 2D grid and is device (and kernel) specific.
# Use deviceQuery (in the nvidia SDK) to help determine this value.
# Otherwise, 65535 is probably sensible (for compute capability >= 2.0).
override MAX_GRID_DIM ?= 65535
# ---------------------------------------------------------------------------

# COMMON FLAGS
override NVCCFLAGS += -Xptxas='-v'
override NVCCFLAGS += -DBLOCK_SIZE=$(BLOCK_SIZE)
override NVCCFLAGS += -DMAX_GRID_DIM=$(MAX_GRID_DIM)
ifneq ($(strip $(TRACE)),)
  override CXXFLAGS  += -DTRACE=$(TRACE)
  override NVCCFLAGS += -DTRACE=$(TRACE)
endif

OBJ = ../common/libcommon.so

all: tpn_half tpn_full

EXTRAINC += -I ../../pairgen/common
EXTRAINC += -I ../../pairgen/tests/hertz
EXTRALIB += -L../common -lcommon
EXTRALIB += -L../../pairgen/tests/hertz -lhertz

tpn_half: driver.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -DHALFNL -DTPN -DPOSIX_TIMER -I $(INCLUDEDIR) $(EXTRAINC) driver.cu -o $@ $(EXTRALIB) -lrt

tpn_full: driver.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS)          -DTPN -DPOSIX_TIMER -I $(INCLUDEDIR) $(EXTRAINC) driver.cu -o $@ $(EXTRALIB) -lrt

clean:
	rm -f tpn_half tpn_full
