#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:29:59 2021

@author: erick
"""

import numpy as np
import matplotlib.pyplot as plt
import functions as f
import easygui as gui

path = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/E_coli/may2021/5/20x_100Hz_05us_EcoliHCB1-07/')

#%%
video = f.videoImport(path, 0)
ni, nj, nk = np.shape(video)

#%%
pi = np.empty(nk)
pj = np.empty(nk)

for k in range(nk):
    p = np.where(video[:,:,k] == video[:,:,k].min())
    pi[k] = p[0][0]
    pj[k] = p[1][0]
    
#%%
plt.imshow(video[:,:,0], cmap='gray'); 
# plt.scatter(pj, pi, marker='_', s=5, c=np.linspace(0, 1, nk))
plt.plot(pi, pj, 'r')
plt.show()

#%%
fig, ax = plt.subplots()
run = True
while run:
    for i in range(nk):
        ax.cla()
        ax.imshow(video[:, :, i], cmap='gray')
        # ax.plot(pi, pj, 'r')
        ax.scatter(pi, pj, s=5, c=np.linspace(0, 1, nk))
        ax.set_title("frame {}".format(i))
        # Note that using time.sleep does *not* work here!
        plt.pause(0.01)
