#!/usr/bin/python -O

import color_scheme
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

cuda_data = np.loadtxt(
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

width = 700
bottom = np.zeros(len(cuda_data['nedge']))

cuda_per_iter_labels = ['aos_gen', 'dev_malloc', 'inverse_map_build']
cuda_bars = []
for i in cuda_per_iter_labels:
  b = ax.bar(cuda_data['nedge'], cuda_data[i], width=width, bottom=bottom, color=color_scheme.colors.next())
  cuda_bars.append(b)
  bottom += cuda_data[i]

op2_data['nedge'] += width
bottom = np.zeros(len(op2_data['nedge']))

op2_per_iter_labels = ['aos_gen', 'op2_init', 'op2_decl', 'op2_plan']
op2_bars = []
for i in op2_per_iter_labels:
  b = ax.bar(op2_data['nedge'], op2_data[i], width=width, bottom=bottom, color=color_scheme.colors.next())
  op2_bars.append(b)
  bottom += op2_data[i]

cuda_bars.reverse()
op2_bars.reverse()
cuda_per_iter_labels.reverse()
op2_per_iter_labels.reverse()

plt.ylim([0,90])
plt.xlabel('Number of contacts')
plt.ylabel('Runtime (milliseconds)')
l1 = plt.legend(map((lambda x: x[0]), cuda_bars), cuda_per_iter_labels, loc=2, title = 'cuda')
l2 = plt.legend(map((lambda x: x[0]), op2_bars), op2_per_iter_labels, loc=2,
    bbox_to_anchor = (0.35,1), title = 'op2')
plt.gca().add_artist(l1)
plt.savefig("breakdown_setup.png")
