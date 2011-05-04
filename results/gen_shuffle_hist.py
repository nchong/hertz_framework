#!/usr/bin/python -O

import numpy as np
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
no_part = op2_data['compute_kernel'][-1]
print "no_part = %f\n" % no_part

data = np.loadtxt(
    "shuffle.data", 
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
           ('result_fetch', 'float'),
           ('seed', 'int')], 
    delimiter=',',
    unpack=True)
hist,bins=np.histogram(data['compute_kernel'],bins=35)
width=0.7*(bins[1]-bins[0])
center=(bins[:-1]+bins[1:])/2

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.bar(center,hist,align='center',width=width,color='#377eb8')
#ax1.axvline(x=no_part, linewidth=5, color='#e41a1c')
plt.xlabel('Runtime (milliseconds)')
plt.ylabel('Frequency')
plt.savefig("op2_shuffle.png")

