#!/bin/bash

cd ../serial; ./run.sh; cd -
cd ../cuda; PROGRAM=./cuda_gpu_timer ./run.sh; cd -
cd ../op2; PROGRAM=./op2_gpu_timer ./run.sh; PROGRAM=./op2_gpu_timer ./part.sh; cd -
cd ../neigh; PROGRAM=./neigh_tpa_gpu_timer OUTPUT=tpa.data ./run.sh;
             PROGRAM=./neigh_bpa_gpu_timer OUTPUT=bpa.data ./run.sh;
             PROGRAM=./neigh_newton_tpa_gpu_timer OUTPUT=tpa.newton.data ./run.sh;
             PROGRAM=./neigh_newton_bpa_gpu_timer OUTPUT=bpa.newton.data ./run.sh; cd -
