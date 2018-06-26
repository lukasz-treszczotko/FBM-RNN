# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 19:00:50 2018

@author: user
"""

from fbm import FBM


f = FBM(n=100, hurst=0.75, length=1, method='daviesharte')

# Generate a fBm realization
fbm_sample = f.fbm()
t_values = f.times()
import matplotlib.pyplot as plt
plt.figure(figsize=(13,6))

def plot(sample):
    values = sample.fbm()
    t_values = sample.times()
    plt.plot(t_values, values)
    plt.title("FBM realization for %d samples and $H=%.2f$" % (sample.n, sample.hurst))
    
f.fbm()
