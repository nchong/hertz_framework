include ../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

override BLOCK_SIZE ?= 32
override MAX_GRID_DIM ?= 65535

override NVCCFLAGS += -Xptxas='-v'
override NVCCFLAGS += -DBLOCK_SIZE=$(BLOCK_SIZE)
override NVCCFLAGS += -DMAX_GRID_DIM=$(MAX_GRID_DIM)

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
	rm -f tpn_half
