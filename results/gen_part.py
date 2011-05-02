#!/usr/bin/python -O

import color_scheme
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

op2_data = np.loadtxt(
    "op2.data", 
    comments='#', 
    dtype=[('nedge', 'float'),
           ('one_time_total','float'), 
           ('per_iter_total', 'float'),
           ('aos_gen', 'float'), 
           ('op2_init', 'float'), 
           ('op2_decl', 'float'),
           ('op2_plan', 'float'), 
           ('compute_kernel', 'float'), 
           ('add_kernel', 'float'), 
           ('result_fetch', 'float')], 
    delimiter=',',
    unpack=True)

data = np.loadtxt(
    "op2_metis.data", 
    comments='#', 
    dtype=[('npart', 'int'),
           ('nedge', 'float'),
           ('one_time_total','float'), 
           ('per_iter_total', 'float'),
           ('aos_gen', 'float'), 
           ('op2_init', 'float'), 
           ('op2_decl', 'float'),
           ('op2_plan', 'float'), 
           ('compute_kernel', 'float'), 
           ('add_kernel', 'float'), 
           ('result_fetch', 'float')], 
    delimiter=',',
    unpack=True)

color_scheme.colors.next()
fig = plt.figure()
ax1 = fig.add_subplot(111)
per_iter_labels = ['compute_kernel', 'add_kernel', 'result_fetch', 'per_iter_total']
for i in per_iter_labels:
  c = color_scheme.colors.next()
  ax1.scatter(data['npart'], data[i], color=c, label=i)
  no_part = op2_data[i][-1]
  ax1.axhline(y=no_part, color=c)
plt.ylim([0,9])
plt.xlabel('Number of partitions')
plt.ylabel('Runtime (milliseconds)')
plt.legend(loc='upper left')
plt.savefig("op2_metis.png")
