#!/bin/bash

PROGRAMS=(../serial/serial_gpu_timer ../cuda/cuda_gpu_timer ../op2/op2_gpu_timer )
OUTPUTS=( serial.data                cuda.data              op2.data )
DATADIR=../data
NUMITER=1000

for ((i=0; i<${#PROGRAMS[@]}; i++)); do
  PROGRAM=${PROGRAMS[$i]}
  OUTPUT=${OUTPUTS[$i]}
  rm -f ${OUTPUT}

  for j in {1..10}; do
    if [ $j == 1 ]; then
      ${PROGRAM} ${DATADIR}/${j}000.step -n ${NUMITER} -v 2> /dev/null | tee -a ${OUTPUT};
    else
      ${PROGRAM} ${DATADIR}/${j}000.step -n ${NUMITER}    2> /dev/null | tee -a ${OUTPUT};
    fi
  done
done
