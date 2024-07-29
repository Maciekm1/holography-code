#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 22:33:22 2022

@author: erick
"""

#%% Import vido and set paramaters
from ssl import PROTOCOL_TLSv1_1
import time
import numpy as np
import easygui
import pandas as pd
import matplotlib.pyplot as plt
import functions as f
import easygui as gui
from multiprocessing import Pool, Process, freeze_support, set_start_method
from multiprocessing import cpu_count
from tqdm import tqdm
from scipy.ndimage import median_filter
from tqdm import tqdm

# PATH = easygui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/')
# PATH = '/media/erick/NuevoVol/LINUX_LAP/PhD/Test/Archea/20x_ArchFM1_30Hz_150us_1_2000frames_every5_300.avi'
PATH = '/media/erick/NuevoVol/LINUX_LAP/PhD/E_coli/June2021/14/sample_1/40x_HCB1_60Hz_1.259us_03/40x_HCB1_60Hz_1.259us_03_430frames.avi'

T0 = time.time()
VID = f.videoImport(PATH, 0)
FRAMES_MEDIAN = 20
I_MEDIAN = f.medianImage(VID, FRAMES_MEDIAN)
I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)
# I_MEDIAN = np.ones((400, 400))

export_csv = True
N = 1.3226
LAMBDA = 0.642              # HeNe
MPP = 40
FS = 0.711                     # Sampling Frequency px/um
psize = 1/FS
SZ = 5                       # # Step size um
NUMSTEPS = 80
# THRESHOLD = 0.5
bandpass = True
med_filter = False

I = VID[:,:,0]
params = (I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, True, False)
# GS = f.modified_propagator(*params)
# RS = f.rayleighSommerfeldPropagator(*params)
# GSS = f.zGradientStack(RS)
# _, BINS = np.histogram(GSS.flatten())
# GSS[GSS < BINS[6]] = 0


# locs1 = f.positions3D(GS, 20, 'None', MPP)
# locs2 = f.positions3D(GSS, 20, 'None', MPP)


#%%
def modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, bandpass, med_filter):
    ## Rayleigh-Sommerfeld Back Propagator
    #   Inputs:          I - hologram (grayscale)
    #             I_MEDIAN - median image
    #                    Z - numpy array defining defocusing distances
    #   Output:        IMM - 3D array representing stack of images at different Z
    import numpy as np
    from functions import bandpassFilter, histeq
    from scipy.ndimage import median_filter

    # Divide by Median image
    I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)
    IN = I / I_MEDIAN

    if med_filter:
        IN = median_filter(IN, size=1)

    # Bandpass Filter
    if bandpass:
        _, BP = bandpassFilter(IN, 2, 30)
        E = np.fft.fftshift(BP) * np.fft.fftshift(np.fft.fft2(IN - 1))
    else:
        E = np.fft.fftshift(np.fft.fft2(IN - 1))

    # Patameter
    LAMBDA = LAMBDA       # HeNe
    FS = FS               # Sampling Frequency px/um
    NI = np.shape(IN)[0]  # Number of rows
    NJ = np.shape(IN)[1]  # Nymber of columns
    Z = SZ*np.arange(0, NUMSTEPS)
    K = 2 * np.pi * N / LAMBDA  # Wavenumber

    # Rayleigh-Sommerfeld Arrays
    jj, ii = np.meshgrid(np.arange(NJ), np.arange(NI))
    const = ((LAMBDA*FS)/(max([NI, NJ])*N))**2
    q = (ii-NI/2)**2 + (jj-NJ/2)**2

    # const = ((LAMBDA*FS)/(max([NI, NJ])*N))**2
    # ff = np.fft.fftfreq(NI, FS)
    # ff = ff**2+ff**2
    # P = const*ff

    P = const*q

    P = np.conj(P)
    Q = np.sqrt(1 - P)-1

    if all(Z > 0):
        Q = np.conj(Q)

    GS = np.empty([NI, NJ, Z.shape[0]], dtype='float32')

    for k in range(Z.shape[0]):
        R = 2*np.pi*1j*q * np.exp(1j*K*Z[k]*Q)
        # GS[:, :, k] = np.abs(1 + np.fft.ifft2(np.fft.ifftshift(E*R)))
        GS[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E*R)))

    _, BINS = np.histogram(GS.flatten())
    GS[GS < BINS[6]] = 0
    # GS[GS < 400] = 0
    
    return GS
 

#%%
X, Y, Z, I_FS, I_GS = [], [] ,[], [], []
for k in tqdm(range(VID.shape[-1])):
    I = VID[:,:,k]
    # T0 = time()
    # IS = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, bandpass, med_filter)
    # GSS = f.zGradientStack(IS)
    # T1 = time()
    # print(T1-T0)
    GS = modified_propagator(VID[:,:,k], I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, bandpass, med_filter)
    # _, BINS = np.histogram(GS.flatten())
    # GS[GS < BINS[6]] = 0
    
    # T2 = time()
    # print(T2-T1)
    # GS[GS < THRESHOLD] = 0
    LOCS = np.empty((1, 3), dtype=object)
    LOCS[0, 0] = f.positions3D(GS, peak_min_distance=20, num_particles='None', MPP=MPP)  # , peak_min_distance, num_particles, MP
    A = LOCS[0, 0].astype('int')
    # LOCS[0, 1] = IM[A[:, 0], A[:, 1], A[:, 2]]
    LOCS[0, 1] = GS[A[:, 0], A[:, 1], A[:, 2]]        #LOCS are in pixels, still need o be converteed to um
    LOCS[0, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]
        
    Y.append(LOCS[0, 0][:, 0] * psize) 
    X.append(LOCS[0, 0][:, 1] * psize)
    Z.append(LOCS[0, 0][:, 2]*SZ)
    I_FS.append(LOCS[0, 1])
    I_GS.append(LOCS[0, 2])

x, y, z, fr = [], [], [], []
for k in range(len(X)):
    for i in range(len(X[k])):
        x.append(X[k][i])
        y.append(Y[k][i])
        z.append(Z[k][i])
        fr.append(k)
        
#%%  3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
%matplotlib inline

fig = pyplot.figure()
ax = Axes3D(fig)
title = '3d plot'
# LOCS = POSITIONS.values
ax.scatter(y, x, z, c=fr, s=25, marker='.')
# ax.plot(LOCS[:, 0], LOCS[:, 1], LOCS[:, 2])
ax.tick_params(axis='both', labelsize=10)
ax.set_title(title, fontsize='20')
ax.set_xlabel('x ($\mu m$)', fontsize='18')
ax.set_ylabel('y ($\mu m$)', fontsize='18')
ax.set_zlabel('z ($\mu m$)', fontsize='18')
pyplot.show()














