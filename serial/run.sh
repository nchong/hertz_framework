#!/bin/bash

if [ -z "${DATADIR}" ]; then
  echo "ERROR: set DATADIR env variable first."
  exit 1
fi

if [ -z "${PROGRAM}" ]; then
  PROGRAM=./serial_cpu_timer
fi

if [ -z "${NUMITER}" ]; then
  NUMITER=1000
fi

if [ -z "${OUTPUT}" ]; then
  OUTPUT=serial.data
fi
rm -f ${OUTPUT}

for j in {1..13}; do
  if [ -z "${CUDA_PROFILE}" ]; then
    export CUDA_PROFILE_LOG="cuda_profile.$j.log"
    rm -f ${CUDA_PROFILE_LOG}
  fi
  if [ $j == 1 ]; then
    ${PROGRAM} ${DATADIR}/${j}000.step -n ${NUMITER} -v 2> /dev/null | tee -a ${OUTPUT};
  else
    ${PROGRAM} ${DATADIR}/${j}000.step -n ${NUMITER}    2> /dev/null | tee -a ${OUTPUT};
  fi
done
