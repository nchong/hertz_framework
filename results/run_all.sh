#!/bin/bash

cd ../serial; ./run.sh; cd -
cd ../cuda; PROGRAM=./cuda_gpu_timer ./run.sh; cd -
cd ../op2; PROGRAM=./op2_gpu_timer ./run.sh; PROGRAM=./op2_gpu_timer ./part.sh; cd -
