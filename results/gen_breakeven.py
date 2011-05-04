#!/usr/bin/python -O

import color_scheme
import math
import numpy as np
import matplotlib.pyplot as plt

def exists(pred, lst):
  return len(filter(pred, lst)) > 0

nedges, serial_total = \
  np.loadtxt("serial.data", comments='#', delimiter=',', usecols=(0,2), unpack=True)
nedges, cuda_setup, cuda_per_iter = \
  np.loadtxt("cuda.data", comments='#', delimiter=',', usecols=(0,1,2), unpack=True)
nedges, op2_setup, op2_per_iter = \
  np.loadtxt("op2.data", comments='#', delimiter=',', usecols=(0,1,2), unpack=True)

cuda_saving = serial_total - cuda_per_iter
cuda_breakeven = map(lambda x: math.ceil(x) if x > 0 else float('nan'), cuda_setup / cuda_saving)
op2_saving = serial_total - op2_per_iter
op2_breakeven = map(lambda x: math.ceil(x) if x > 0 else float('nan'), op2_setup / op2_saving)

color_scheme.colors.next()
fig = plt.figure()
ax1 = fig.add_subplot(111)
if (exists(lambda x: not math.isnan(x), cuda_breakeven)):
  ax1.scatter(nedges, cuda_breakeven, color=color_scheme.colors.next(), label="cuda")
else:
  print "No breakeven points for cuda."
  print "serial_total", serial_total
  print "cuda_setup", cuda_setup
  print "cuda_per_iter", cuda_per_iter
  print "cuda_saving", cuda_saving
  print "cuda_breakeven", cuda_breakeven
if (exists(lambda x: not math.isnan(x), op2_breakeven)):
  ax1.scatter(nedges, op2_breakeven,  color=color_scheme.colors.next(), label="op2")
else:
  print "No breakeven points for op2."
  print "serial_total", serial_total
  print "op2_setup", op2_setup
  print "op2_per_iter", op2_per_iter
  print "op2_saving", op2_saving
  print "op2_breakeven", op2_breakeven

plt.ylim([0,150])
plt.xlabel('Number of contacts')
plt.ylabel('Number of iterations')
plt.legend(loc='upper left')
plt.savefig("breakeven.png")
