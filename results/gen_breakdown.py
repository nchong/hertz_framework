#!/usr/bin/python -O

import color_scheme
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

data = np.loadtxt(
    "cuda.data", 
    comments='#', 
    dtype=[('nedge', 'float'),
           ('one_time_total','float'), 
           ('per_iter_total', 'float'),
           ('aos_gen', 'float'), 
           ('dev_malloc', 'float'), 
           ('inverse_map_build', 'float'),
           ('aos_memcpy', 'float'), 
           ('compute_kernel', 'float'), 
           ('gather_kernel', 'float'), 
           ('result_fetch', 'float')], 
    delimiter=',',
    unpack=True)

width = 1500
bottom = np.zeros(len(data['nedge']))
per_iter_labels = ['aos_memcpy', 'compute_kernel', 'gather_kernel', 'result_fetch']
bars = []
for i in per_iter_labels:
  b = plt.bar(data['nedge'], data[i], width=width, bottom=bottom, color=color_scheme.colors.next())
  bars.append(b)
  bottom += data[i]
plt.ylim([0,6])
plt.xlabel('Number of contacts')
plt.ylabel('Runtime (milliseconds)')
plt.legend(map((lambda x: x[0]), bars), per_iter_labels, loc='upper left')
plt.savefig("cuda_breakdown.png")

plt.clf()
color_scheme.colors.next()
color_scheme.colors.next()
color_scheme.colors.next()
color_scheme.colors.next()

data = np.loadtxt(
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
width = 1500
bottom = np.zeros(len(data['nedge']))
per_iter_labels = ['compute_kernel', 'add_kernel', 'result_fetch']
bars = []
for i in per_iter_labels:
  b = plt.bar(data['nedge'], data[i], width=width, bottom=bottom, color=color_scheme.colors.next())
  bars.append(b)
  bottom += data[i]
plt.ylim([0,6])
plt.xlabel('Number of contacts')
plt.ylabel('Runtime (milliseconds)')
plt.legend(map((lambda x: x[0]), bars), per_iter_labels, loc='upper left')
plt.savefig("op2_breakdown.png")
