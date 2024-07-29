#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:27:07 2021

@author: erick
"""

import numpy as np
import matplotlib.pyplot as plt
import functions as f
import easygui as gui

path = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/Thesis/')
VID = f.videoImport(path, 0)
# VID, cdf = f.histeq(VID)
I = VID[:, :, 0] #133
# I, cdf = f.histeq(IN)

I_MEDIAN = f.medianImage(VID, 20)
# I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)

# IN = I / I_MEDIAN
# IN, cdf = f.histeq(IN)

# path_med = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/E_coli/may2021/5/')
# I_MEDIAN = np.ones_like(I)
# I_MEDIAN = plt.imread(path_med)
N = 1.3226
LAMBDA = 0.642              # HeNe
MPP = 40
FS = 0.711 * (MPP/10)                     # Sampling Frequency px/um
SZ = 2.5  

rs = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, 70, True, False)
# gs = f.zGradientStack(rs)
# threshold = 0.1
# gs[gs < threshold] = 0
# locs = f.positions3D(gs, peak_min_distance=20, num_particles='None', MPP=MPP)

# plt.imshow(rs[:, :, 0], cmap='gray')
# plt.plot(locs[:, 1], locs[:, 0], 'r.')

f.imshow_slider(rs, 2, 'gray')

# _, BINS = np.histogram(rs.flatten())
# rs[rs < BINS[8]] = 0

#%%
fr = np.array([0, 10, 15, 20])*SZ

plt.imsave('archea_gs_100.png', gs[:, :, 20], cmap='gray')

f
#%%
path_write = gui.diropenbox()

rs_binary = np.empty_like(rs)
for i in range(rs_binary.shape[-1]):
    temp = rs[:, :, i]
    t = np.zeros_like(temp)
    t[temp >= temp.mean()] = 255
    rs_binary[:, :, i] = t
    np.uint8(rs_binary)
    plt.imsave(path_write+'\\'+str(i)+'.png', rs_binary[ci-D:ci+D, cj-D:cj+D , i], cmap='gray')

#%% Zero mean
path_write = gui.diropenbox()

#% To write VID ZM
# for k, im in enumerate(img):
#     temp = im
#     mn, std = temp.mean(), temp.std()
#     temp_zm = (temp - mn)/std
#     b = temp_zm > 0
#     temp_zm[temp_zm < 0] = 0
#     eq, _ = f.histeq(temp_zm)
#     # rs_zm[:, :, k] = eq
#     plt.imsave(path_write+'\\'+str(k)+'.png',eq, cmap='gray')
    
#%

rs_zm = np.empty_like(rs)
for k in range(rs.shape[-1]):
    temp = rs[:, :, k]
    mn, std = temp.mean(), temp.std()
    temp_zm = (temp - mn)/std
    b = temp_zm > 0
    temp_zm[temp_zm < 0] = 0
    eq, _ = f.histeq(temp_zm)
    rs_zm[:, :, k] = eq
    plt.imsave(path_write+'\\'+str(k)+'.png', rs_zm[ci-D:ci+D, cj-D:cj+D , k], cmap='gray')

#%% Export LUT

path_write = gui.diropenbox()

ci, cj = 220, 261
D = 80
modes = ['Normal', 'Binary', 'ZM']
mode = modes[1]

# for i in range(rs.shape[-1]):
for i in range(rs.shape[-1]):
    # print(i)
    if mode == 'Normal':
        plt.imsave(path_write+'\\'+str(i)+'.png', rs[ci-D:ci+D, cj-D:cj+D , i], cmap='gray') 
    elif mode == 'Binary':
        plt.imsave(path_write+'\\'+str(i)+'.png', rs_binary[ci-D:ci+D, cj-D:cj+D , i], cmap='gray') 
    elif mode == 'ZM':
        plt.imsave(path_write+'\\'+str(i)+'.png', rs_zm[ci-D:ci+D, cj-D:cj+D , i], cmap='gray')
        
        
    
    
    