#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 5 19:36:33 2020

@author: erick
"""

#%%
import matplotlib as mpl
import mpld3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
mpl.rc('figure',  figsize=(10, 6))
import functions as f
import sklearn.cluster as cl
import time
import easygui as gui
import hdbscan
from scipy.optimize import linear_sum_assignment
from hungarian_algorithm import algorithm

PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/')
# DF = pd.read_csv(PATH, index_col=0)
DF = pd.read_csv(PATH)
DF = DF.head(1000000)

# DF = pd.read_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_rayleighSommerfeld_Results.csv', index_col=0)
# DF= pd.read_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv', index_col=0)
# DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1_500_FRAMES_MODIFIED.csv', index_col=0)
# DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_2001_FRAMES_MODIFIED.csv', index_col=0)
# DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_2001_FRAMES_RS_TH019.csv', index_col=0)

DF.index = np.arange(len(DF))
# D = DF[['X', 'Y', 'Z', 'FRAME']].values
D = DF.values
# DD = A[:, :3]

# method = 'KMeans'
# method = 'DBSCAN'
plot_results = False


#%% K-Means for track detection
DF_KMeans = DF.copy()
T0_kmeans = time.time()
kmeans = cl.KMeans(n_clusters=D[D[:, 5] == 0].shape[0], init='k-means++').fit(DF_KMeans[['X', 'Y', 'Z']])
DF_KMeans['PARTICLE'] = kmeans.labels_
LINKED_KMeans = DF_KMeans
T_kmeans = time.time() -  T0_kmeans
PARTICLES_KMeans = LINKED_KMeans['PARTICLE'].unique().shape[0]
print('KMeans',T0_kmeans, T0_kmeans + T_kmeans, T_kmeans)


if plot_results == True:
    from mpl_toolkits.mplot3d import Axes3D
    # %matplotlib qt
    fig1 = plt.figure(1)
    ax1 = Axes3D(fig1)
    A = LINKED_KMeans.__array__()
    A = LINKED_KMeans[LINKED_KMeans.PARTICLE != -1].values
    p1 = ax1.scatter(A[:, 0], A[:, 1], A[:, 2], s=1, marker='o', c=A[:, 6])
    fig1.colorbar(p1)
    plt.show(fig1)


# Smooth trajectories
# LINKED = LINKED[LINKED.PARTICLE != -1]

# For Archea data that looks messy
value_counts = LINKED_KMeans.PARTICLE.value_counts()
values = value_counts[value_counts.values < 1000]
idx = LINKED_KMeans.PARTICLE.isin(values.index)
LINKED_KMeans = LINKED_KMeans.iloc[idx.values]
particle_num = LINKED_KMeans.PARTICLE.unique()

T0_smooth_KMeans = time.time()
smoothed_curves = -np.ones((1, 4))
for pn in particle_num:
    L = LINKED_KMeans[LINKED_KMeans.PARTICLE == pn].values
    X = f.smooth_curve(L)
    
    if X != -1:
        smoothed_curves = np.vstack((smoothed_curves, np.stack((X[0], X[1], X[2], pn*np.ones_like(X[1])), axis=1))) 

smoothed_curves = smoothed_curves[1:, :]
smoothed_curves_df_KMeans = pd.DataFrame(smoothed_curves, columns=['X', 'Y' ,'Z', 'PARTICLE'])
T_smooth_KMeans = time.time() - T0_smooth_KMeans
print('Smooth KMeans',T0_smooth_KMeans, T0_smooth_KMeans + T_smooth_KMeans, T_smooth_KMeans)



#%% Density Based Spatial Clustering (DBSC)
DF_DBSCAN = DF.copy()
T0_DBSCAN = time.time()
DBSCAN = cl.DBSCAN(eps=5, min_samples=8).fit(DF_DBSCAN[['X', 'Y', 'Z']])
DF_DBSCAN['PARTICLE'] = DBSCAN.labels_
LINKED_DBSCAN = DF_DBSCAN
L = LINKED_DBSCAN.drop(np.where(LINKED_DBSCAN.PARTICLE.values == -1)[0])
T_DBSCAN = time.time() - T0_DBSCAN
PARTICLES_DBSCAN = LINKED_DBSCAN['PARTICLE'].unique().shape[0] - 1
print('DBSCAN',T0_DBSCAN, T0_DBSCAN + T_DBSCAN, T_DBSCAN)

if plot_results == True:
    from mpl_toolkits.mplot3d import Axes3D
    # %matplotlib qt
    fig2 = plt.figure(2)
    ax2 = Axes3D(fig2)
    A = LINKED_DBSCAN.__array__()
    A = LINKED_DBSCAN[LINKED_DBSCAN.PARTICLE != -1].values
    p2 = ax2.scatter(A[:, 0], A[:, 1], A[:, 2], s=1, marker='o', c=A[:, 6])
    fig2.colorbar(p2)
    plt.show(fig2)


# Smooth trajectories
# LINKED = LINKED[LINKED.PARTICLE != -1]

# For Archea data that looks messy
value_counts = LINKED_DBSCAN.PARTICLE.value_counts()
values = value_counts[value_counts.values < 1000]
idx = LINKED_DBSCAN.PARTICLE.isin(values.index)
LINKED_DBSCAN = LINKED_DBSCAN.iloc[idx.values]
particle_num = LINKED_DBSCAN.PARTICLE.unique()

T0_smooth_DBSCAN = time.time()
smoothed_curves = -np.ones((1, 4))
for pn in particle_num:
    L = LINKED_DBSCAN[LINKED_DBSCAN.PARTICLE == pn].values
    X = f.smooth_curve(L)
    
    if X != -1:
        smoothed_curves = np.vstack((smoothed_curves, np.stack((X[0], X[1], X[2], pn*np.ones_like(X[1])), axis=1))) 

smoothed_curves = smoothed_curves[1:, :]
smoothed_curves_df_DBSCAN = pd.DataFrame(smoothed_curves, columns=['X', 'Y' ,'Z', 'PARTICLE'])
T_smooth_DBSCAN = time.time() - T0_smooth_DBSCAN
print('Smooth DBSCAN',T0_smooth_DBSCAN, T0_smooth_DBSCAN + T_smooth_DBSCAN, T_smooth_DBSCAN)



#%% Hungrian algorithm
DF_HA = DF.copy()
T0_HA = time.time()
FRAME_DATA = np.empty(int(DF_HA['FRAME'].max()), dtype=object)
SIZE = np.empty(FRAME_DATA.shape[0])
for i in range(FRAME_DATA.shape[0]):
    FRAME_DATA[i] = D[D[:, 5] == i]
    SIZE[i] = FRAME_DATA[i].shape[0]    

# particle_index = np.empty((int(SIZES[0]), len(FRAME_DATA)), dtype='int')
TRACKS = [FRAME_DATA[0]]
sizes = np.empty(len(FRAME_DATA), dtype='int')
sizes[0] = len(FRAME_DATA[0])
for k in range(len(FRAME_DATA)-1):  
    # print(k)
    
    if k==0:    
        R1 = FRAME_DATA[k]
        R2 = FRAME_DATA[k+1]
    else:
        R1 = TRACKS[k]
        R2 = FRAME_DATA[k+1]
    
    # COST = np.empty((int(SIZE.max()), int(SIZE.max())))
    COST = np.empty((len(R1), len(R2)))
    # for i in range(int(SIZE.max())):
    #     for j in range(int(SIZE.max())):        
            
    for i in range(len(R1)):
        for j in range(len(R2)):
            
            COST[i, j] = np.sqrt(sum((R1[i, :3]- R2[j, :3])**2))      
    # row_ind, particle_index[:, k] = linear_sum_assignment(COST)
    row_ind, particle_index = linear_sum_assignment(COST)
    sizes[k+1] = len(particle_index)
    # print(len(particle_index))
    # TRACKS.append(R2[particle_index[:, k], :])
    TRACKS.append(R2[particle_index, :])

# print(time.time() - T0)

#%
TRACK = np.vstack(TRACKS)
TR = pd.DataFrame(TRACK, columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME'])
# TR['PARTICLE'] = np.tile(np.arange(SIZES[0]), len(TRACKS))

particle_column = []
for i in range(len(sizes)):
    particle_column.append(np.arange(sizes[i]))    

particle_column = np.concatenate(particle_column, axis=0)
TR['PARTICLE'] = particle_column
LINKED_HA = TR
T_HA = time.time() - T0_HA
PARTICLES_HA = LINKED_HA['PARTICLE'].unique().shape[0]
print('HA',T0_HA, T0_HA + T_HA, T_HA)

if plot_results == True:
    from mpl_toolkits.mplot3d import Axes3D
    # %matplotlib qt
    fig3 = plt.figure(3)
    ax3 = Axes3D(fig3)
    A = LINKED_HA.__array__()
    A = LINKED_HA[LINKED_HA.PARTICLE != -1].values
    p3 = ax3.scatter(A[:, 0], A[:, 1], A[:, 2], s=1, marker='o', c=A[:, 6])
    fig3.colorbar(p3)
    plt.show(fig3)



# Smooth trajectories
# LINKED = LINKED[LINKED.PARTICLE != -1]

# For Archea data that looks messy
value_counts = LINKED_HA.PARTICLE.value_counts()
values = value_counts[value_counts.values < 1000]
idx = LINKED_HA.PARTICLE.isin(values.index)
LINKED_HA = LINKED_HA.iloc[idx.values]
particle_num = LINKED_HA.PARTICLE.unique()

T0_smooth_HA = time.time()
smoothed_curves = -np.ones((1, 4))
for pn in particle_num:
    L = LINKED_HA[LINKED_HA.PARTICLE == pn].values
    X = f.smooth_curve(L)
    
    if X != -1:
        smoothed_curves = np.vstack((smoothed_curves, np.stack((X[0], X[1], X[2], pn*np.ones_like(X[1])), axis=1))) 

smoothed_curves = smoothed_curves[1:, :]
smoothed_curves_df_HA = pd.DataFrame(smoothed_curves, columns=['X', 'Y' ,'Z', 'PARTICLE'])
T_smooth_HA = time.time() - T0_smooth_HA
print('Smooth HA',T0_smooth_HA, T0_smooth_HA + T_smooth_HA, T_smooth_HA)


#%% Create runtime array
TIME_LIST =np.reshape(np.array([T_kmeans,
            PARTICLES_KMeans,
            T_smooth_KMeans,
            T_DBSCAN,
            PARTICLES_DBSCAN,
            T_smooth_DBSCAN,
            T_HA,
            PARTICLES_HA,
            T_smooth_HA]), (1, 9))

print(TIME_LIST)

TIME_LIST_DF = pd.DataFrame(data = TIME_LIST)





