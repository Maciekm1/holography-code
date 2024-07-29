# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:27:14 2022

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

#%% 
slices = [0, 78, 100, 122, 137, 157, 173, 216, 226, 244, 261]
expath = 'C:\\Users\\eers500\\Documents\\PhD\\E_coli\\may2021\\5\\20x_100Hz_05us_EcoliHCB1_07\\NEW ANALYSIS\\Accuracy experiment\\'

for i, s in enumerate(slices):
    im = VID[:, :, s]
    plt.imsave(expath+str(i)+'.png', im, cmap='gray')
    