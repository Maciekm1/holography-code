#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:48:58 2022

@author: erick
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f

path = '/media/erick/NuevoVol/LINUX_LAP/PhD/Archea_LW/modified_propagator/20x_ArchFM1_30Hz_150us_1_2000frames_every5_300-1-1_v2.csv'

data = pd.read_csv(path, header=None).values

gsl = np.empty((300, 300, 100))
for k in range(100):
    # gsl[:, :, k] = data[k*300:k*300+300, :] / np.max(data[k*300:k*300+300, :])
    gsl[:, :, k] = data[k*300:k*300+300, :] 

# _, BINS = np.histogram(gsl.flatten())
# gsl[gsl < BINS[6]] = 0
#%%
I = plt.imread('/media/erick/NuevoVol/LINUX_LAP/PhD/Archea_LW/modified_propagator/20x_ArchFM1_30Hz_150us_1_2000frames_every5_300-1-1.png')
I_MEDIAN = plt.imread('/media/erick/NuevoVol/LINUX_LAP/PhD/Archea_LW/modified_propagator/MED_20x_ArchFM1_30Hz_150us_1_2000frames_every5_300-1.png')

N = 1.3226
LAMBDA = 0.642              # HeNe
MPP = 20
FS = 0.711                     # Sampling Frequency px/um
psize = 1/FS
SZ = 1                       # # Step size um
NUMSTEPS = 100
bandpass = False
med_filter = False

gse = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, bandpass, med_filter)
# for k in range(100):
#     gse[:, :, k] = gse[:, :, k] / np.max(gse[:, :, k])

# _, BINS = np.histogram(gse.flatten())
# gse[gse < BINS[6]] = 0

#%%
plt.figure()
plt.subplot(1,2,1)
plt.imshow(gsl[:, :, 1], cmap='gray')

plt.subplot(1,2,2)
plt.imshow(gse[:, :, 1], cmap='gray')

plt.figure()
plt.subplot(1,2,1)
plt.imshow(gsl[:, :, 50], cmap='gray')

plt.subplot(1,2,2)
plt.imshow(gse[:, :, 50], cmap='gray')