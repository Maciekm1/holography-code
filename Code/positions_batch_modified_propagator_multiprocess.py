#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 13:21:30 2021

@author: erick
"""

#%% Import video and set paramaters
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

# PATH = easygui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD')
# PATH = '/media/erick/NuevoVol/LINUX_LAP/PhD/Test/Archea/20x_ArchFM1_30Hz_150us_1_2000frames_every5_300.avi'
# PATH = '/media/erick/NuevoVol/LINUX_LAP/PhD/E_coli/June2021/14/sample_1/40x_HCB1_60Hz_1.259us_03/40x_HCB1_60Hz_1.259us_03_430frames.avi'
# PATH = '/media/erick/NuevoVol/LINUX_LAP/PhD/E_coli/may2021/5/20x_100Hz_05us_EcoliHCB1-05/20x_100Hz_05us_EcoliHCB1-05.avi'
PATH = '/media/erick/NuevoVol/LINUX_LAP/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames.avi'


T0 = time.time()
VID = f.videoImport(PATH, 0)
FRAMES_MEDIAN = 20
I_MEDIAN = f.medianImage(VID, FRAMES_MEDIAN)
I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)
# I_MEDIAN = np.ones((400, 400))

export_csv = False
# N = 1.3226
# LAMBDA = 0.642              # HeNe
# # MPP = 10
# FS = 0.711                     # Sampling Frequency px/um
# SZ = 5                       # # Step size um
# NUMSTEPS = 150
# THRESHOLD = 0.1

#%%  Calculate propagators, gradient stack and compute particle position ins 3D
NUM_FRAMES = np.shape(VID)[-1]
# NUM_FRAMES = int(np.floor(np.shape(VID)[-1]/2))
# NUM_FRAMES = 100
# LOCS = np.empty((NUM_FRAMES, 3), dtype=object)
# INTENSITY = np.empty(NUM_FRAMES, dtype=object)

pbar = tqdm(total=NUM_FRAMES/cpu_count())

def my_function(TUPLE):
    I = TUPLE[0]
    I_MEDIAN = TUPLE[1]
    # I_MEDIAN = np.load('/media/erick/NuevoVol/LINUX_LAP/PhD/Colloids/MED_20x_50Hz_100us_642nm_colloids-1.npy')
    N = 1.3226
    LAMBDA = 0.642              # HeNe
    MPP = 10
    FS = 0.711 * (MPP/10)                  # Sampling Frequency px/um
    psize = 1/FS
    SZ = 5                       # # Step size um
    NUMSTEPS = 80
    bandpass = True
    med_filter = False
    X, Y, Z, I_FS, I_GS = [], [] ,[], [], []
    # X, Y, Z, I_FS, I_GS = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
#
    # for i in range(NUM_FRAMES):
    # I = IT[i]
    # IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, True).astype('float32')
    # GS = f.zGradientStack(IM).astype('float32')  
    # GS = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS)  # Modified propagator
    GS = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, bandpass, med_filter)
      
    # GS[GS < THRESHOLD] = 0
    LOCS = np.empty((1, 3), dtype=object)
    LOCS[0, 0] = f.positions3D(GS, peak_min_distance=20, num_particles='None', MPP=MPP)  # , peak_min_distance, num_particles, MP
    A = LOCS[0, 0].astype('int')
    # LOCS[0, 1] = IM[A[:, 0], A[:, 1], A[:, 2]]
    LOCS[0, 1] = GS[A[:, 0], A[:, 1], A[:, 2]]        #LOCS are in pixels, still need o be converteed to um
    LOCS[0, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]
        
    Y.append(LOCS[0, 0][:, 1] * psize) 
    X.append(LOCS[0, 0][:, 0] * psize)
    Z.append(LOCS[0, 0][:, 2]*SZ)
    I_FS.append(LOCS[0, 1])
    I_GS.append(LOCS[0, 2])
    
    #print(time.time())
        
    pbar.update(1)
        
    return X, Y, Z, I_FS, I_GS

    
IT = np.empty((NUM_FRAMES), dtype=object)
MED = np.empty((NUM_FRAMES), dtype=object)
for i in range(NUM_FRAMES):
    IT[i] = VID[:, :, i]
    MED[i] = I_MEDIAN

if __name__ == "__main__":    
    # freeze_support()
    # set_start_method('spawn')
    T0 = time.time()
    pool = Pool(cpu_count())
    #pool = Pool(1)
    results = pool.map(my_function, zip(IT, MED))
    # results = pool.map_async(my_function, zip(IT, MED))
    # results = pool.imap_unordered(my_function, zip(IT, MED))
    # results = pool.apply_async(my_function, zip(IT, MED), callback=update)
    pool.close()  # 'TERM'
    # pool.join()   # 'KILL'
    T = time.time() - T0
    print('\n Elapsed time: ', T)

    POSITIONS = pd.DataFrame(columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME'])
    for i in range(NUM_FRAMES):
      # XYZ, I_FS, I_GS, FRAME = LOCS[i, 0], LOCS[i, 1], LOCS[i, 2], i*np.ones_like(LOCS[i, 2])
            
        X, Y, Z, I_FS, I_GS, FRAME = results[i][0][0], results[i][1][0], results[i][2][0], results[i][3][0], results[i][4][0], i*np.ones_like(results[i][0][0])
        DATA = np.concatenate((np.expand_dims(X, axis=1), np.expand_dims(Y, axis=1), np.expand_dims(Z, axis=1), np.expand_dims(I_FS, axis=1), np.expand_dims(I_GS, axis=1), np.expand_dims(FRAME, axis=1)), axis=1)
        POSITIONS = POSITIONS.append(pd.DataFrame(DATA, columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME']))

    if export_csv:        
        # EXPORT_PATH = PATH[:-4]+'_TH01_MPD35_multiprocess.csv'
        EXPORT_PATH = gui.filesavebox(filetypes='csv', default='.csv', msg='Save file')
        POSITIONS.to_csv(EXPORT_PATH)  # For leptospira data
	
#%%  3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
# %matplotlib qt 

#ax = Axes3D(fig)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

title = '3d plot'
LOCS = POSITIONS.values
ax.scatter(LOCS[:, 0], LOCS[:, 1], LOCS[:, 2], s=25, c=LOCS[:, 5], marker='.')
# ax.plot(LOCS[:, 0], LOCS[:, 1], LOCS[:, 2])
ax.tick_params(axis='both', labelsize=10)
ax.set_title(title, fontsize='20')
ax.set_xlabel('x ($\mu m$)', fontsize='18')
ax.set_ylabel('y ($\mu m$)', fontsize='18')
ax.set_zlabel('z ($\mu m$)', fontsize='18')
plt.show()

   
