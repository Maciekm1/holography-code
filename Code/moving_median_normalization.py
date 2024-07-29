# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:40:34 2022

@author: eers500
"""

import numpy as np
import matplotlib.pyplot as plt
import functions as f
import easygui as gui
from tqdm import tqdm

path = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/E_coli/may2021/5/')
VID = f.videoImport(path, 0)
ni, nj, nk = np.shape(VID)


#%% Normalize with median image of full video
vidn = np.empty_like(VID)
med = f.medianImage(VID, 20)
med[med == 0] = np.mean(med)

for k in tqdm(range(nk)):
    vidn[:, :, k] = VID[:, :, k] / med

#%% Normalize with moving mean image
nframes = 20
vid_norm = np.empty_like(VID)

for k in tqdm(range(nk)):
    
    if k <= nframes:
        median = np.median(VID[:, :, :nframes], axis=2)
        median[median == 0] = np.mean(median)
        vid_norm[:, :, k] = VID[:, :, k] / median
        
    elif k >= nk-nframes:
        median = np.median(VID[:, :, nk-nframes:nk], axis=2)
        median[median == 0] = np.mean(median)
        vid_norm[:, :, k] = VID[:, :, k] / median
        
    else:
        median = np.median(VID[:, :, k-nframes:k+nframes], axis=2)
        median[median == 0] = np.mean(median)
        vid_norm[:, :, k] = VID[:, :, k] / median


#%%

fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
ax[0, 0].imshow(vidn[:, :, 50], cmap='gray')
ax[0, 1].imshow(vid_norm[:, :, 50], cmap='gray')

ax[1, 0].imshow(vidn[:, :, 200], cmap='gray')
ax[1, 1].imshow(vid_norm[:, :, 200], cmap='gray')

ax[2, 0].imshow(vidn[:, :, 270], cmap='gray')
ax[2, 1].imshow(vid_norm[:, :, 270], cmap='gray')