#!/bin/bash

for i in {0..1000}; do echo -n "$i, " | tee -a shuf.data; ../op2/op2_gpu_timer ../data/10000.step -s $i 2> /dev/null | tee -a shuf.data; done
