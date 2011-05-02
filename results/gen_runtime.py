#!/usr/bin/python -O

import color_scheme
import numpy as np
import matplotlib.pyplot as plt

nedges, serial = \
  np.loadtxt("serial.data", comments='#', delimiter=',', usecols=(0,2), unpack=True)
nedges, cuda = \
  np.loadtxt("cuda.data", comments='#', delimiter=',', usecols=(0,2), unpack=True)
nedges, op2 = \
  np.loadtxt("op2.data", comments='#', delimiter=',', usecols=(0,2), unpack=True)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(nedges, serial, color=color_scheme.colors.next(), label="serial")
ax1.scatter(nedges, cuda,   color=color_scheme.colors.next(), label="cuda")
ax1.scatter(nedges, op2,    color=color_scheme.colors.next(), label="op2")

plt.xlabel('Number of contacts')
plt.ylabel('Runtime (milliseconds)')
plt.legend(loc='upper left')
plt.savefig("runtime.png")
