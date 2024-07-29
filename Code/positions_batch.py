# -*- coding: utf-8 -*-
"""Calculate position of particles for several frames"""
#%% Import video and set paramaters
import time
import os
import numpy as np
import easygui as gui
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib qt
import functions as f
from progress.bar import Bar


PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/')
# PATH = 'MF1_30Hz_200us_awaysection.avi'
#PATH = '10x_laser_50Hz_10us_g1036_bl1602_500frames.avi'
# PATH = '/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames.avi'
# PATH = '/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1.avi'
# THRESHOLD = gui.enterbox(msg='Threshold for Gradient Stack', title='Threshold', default='0.1')

# OPTIONS = gui.multenterbox(msg='Threshold and file export',
#                             title='Thresold and export',
#                             fields=['THRESHOLD for GS:',
#                                    'Number of frames for calculations:',
#                                    'Export CSV file? (y/n):',
#                                    'Peak min distance for GS Z-Projection: (~20, ~35)', # 35 for colloid data, 20 for everything else
#                                    'Number of steps (150, 40, etc)']) 

# THRESHOLD, NUM_FRAMES, EXPORT, PEAK_MIN_DISTANCE, NUMSTEPS, CALC_MEDIAN = gui.multenterbox(msg='Threshold and file export',
OPTIONS = gui.multenterbox(msg='Threshold and file export',
                            title='Thresold and export',
                            fields=['THRESHOLD for GS:',
                                   'Number of frames for calculations:',
                                   'Export CSV file? (y/n):',
                                   'Peak min distance for GS Z-Projection: (~20, ~35)', # 35 for colloid data, 20 for everything else
                                   'Number of steps (150, 40, etc)',
                                   'Calculate Median Image? (y/n)']) 

THRESHOLD = OPTIONS[0]
NUM_FRAMES = OPTIONS[1]
EXPORT = OPTIONS[2]
PEAK_MIN_DISTANCE = OPTIONS[3]

NUMSTEPS = OPTIONS[4]
CALC_MEDIAN = OPTIONS[5]

# Import time
T0 = time.time()
vid = f.videoImport(PATH, 0)
T_video_import = time.time() - T0

# Median Image
T0_median_image = time.time()
if CALC_MEDIAN == 'y':
    FRAMES_MEDIAN = 20
    I_MEDIAN = f.medianImage(vid, FRAMES_MEDIAN)
    
elif CALC_MEDIAN == 'n':
    I_MEDIAN = np.ones((np.shape(vid)[0], np.shape(vid)[1]))
    
T_median_image = time.time() - T0_median_image

N = 1.3226
LAMBDA = 0.642              # HeNe
MPP = 20 
FS = 0.711                     # Sampling Frequency px/um
SZ = 5                        # Step size um
# NUMSTEPS = int(OPTIONS[4])
# THRESHOLD = np.float(OPTIONS[0])
# PEAK_MIN_DISTANCE = int(OPTIONS[3])

THRESHOLD = np.fromstring(THRESHOLD, sep=' ')
NUM_FRAMES = np.fromstring(NUM_FRAMES, sep=' ').astype(int)
PEAK_MIN_DISTANCE = np.fromstring(PEAK_MIN_DISTANCE, sep=' ').astype(int)
NUMSTEPS = np.fromstring(NUMSTEPS, sep=' ').astype(int)


#% Test positions3D
# FRAME = 20
# I = VID[:, :, FRAME]
# IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, MPP, FS, SZ, NUMSTEPS)
# GS = f.zGradientStack(IM)  # GradientStack and RS propagator
# GS[GS < THRESHOLD] = 0
# LOCS = f.positions3D(GS)

# f.plot3D(LOCS, title='pos')

#%  Calculate propagators, gradient stack and compute particle position ins 3D
# if np.float(OPTIONS[1]) == -1:
#     NUM_FRAMES = np.shape(VID)[-1]
# else:
#     NUM_FRAMES = int(OPTIONS[1])

PARAMETER_LIST = []
for j in range(len(THRESHOLD)):

    # NUM_FRAMES = 5
    VID = vid[:, :, :NUM_FRAMES[j]]
    LOCS = np.empty((NUM_FRAMES[j], 3), dtype=object)
    # INTENSITY = np.empty(NUM_FRAMES, dtype=object)
    
    # IMM = np.empty((512, 512, 150, NUM_FRAMES), dtype='float16')
    # GSS = np.empty((512, 512, 150, NUM_FRAMES), dtype='float16')
    
    T = []
    T_RS = []
    T_GS = []
    T_positions3D = []
    T0_loop = time.time()
    bar = Bar('Processing', max=NUM_FRAMES[j], suffix='%(percent).1f%% - %(eta)ds')
    
    for i in range(NUM_FRAMES[j]):
        I = VID[:, :, i]
        
        T0_RS = time.time()
        IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS[j], True).astype('float32')
        T_RS.append(time.time() - T0_RS)
    
        # IMM[:, :, :, i] = IM
        
        T0_GS = time.time()
        GS = f.zGradientStack(IM).astype('float32')  
        GS[GS < THRESHOLD[j]] = 0
        T_GS.append(time.time() - T0_GS)
        # GS = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS)  # Modified propagator
        # GSS[:, :, :, i] = GS
            
        # Turn to zero 99.99% of values
        # HIST, BINS = np.histogram(GSS.flatten(), 1000)
        # CDF = np.cumsum(HIST)
        # ID = np.where(CDF <= CDF.max()*0.9999)   
        # GS[GS <= BINS[ID[0][-1]]] = 0
        T0_positions3D = time.time()
        LOCS[i, 0] = f.positions3D(GS, peak_min_distance=PEAK_MIN_DISTANCE[j],  num_particles=1, MPP=MPP)
        T_positions3D.append(time.time() - T0_positions3D)
    
        A = LOCS[i, 0].astype('int')
        LOCS[i, 1] = IM[A[:, 0], A[:, 1], A[:, 2]]
        LOCS[i, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]
        T.append(time.time()-T0_loop)
        print(str(i+1)+' of '+ str(NUM_FRAMES[j]), j+1, str(len(THRESHOLD)), (time.time()-T0_loop))
        bar.next()
    bar.finish()
    T_total = time.time() - T0
    T_total = T[-1]
    print(T_total/60)
    
    EXPORT_PATH = PATH[:-4]+'_'+np.str(NUM_FRAMES[j])+'_FRAMES_RS_TH'+np.str(THRESHOLD[j]).replace('.','')+'_MPD'+np.str(PEAK_MIN_DISTANCE[j])+'_SZ'+np.str(SZ)+'_STPS'+np.str(NUMSTEPS[j])+'.csv'
    
    # TIME = {'number_frames': NUM_FRAMES,
    #         'num_steps': NUMSTEPS,
    #         'step_size': SZ,
    #         'threshold_GS': THRESHOLD,
    #         'video_import': T_video_import,
    #         'median_image': T_median_image,
    #         'RS': np.mean(T_RS),
    #         'GS': np.mean(T_GS),
    #         'positions3D': np.mean(T_positions3D),
    #         'Total_runtime': T_total}
    
    TIME_LIST = [NUM_FRAMES[j],
                 NUMSTEPS[j],
                 SZ,
                 PEAK_MIN_DISTANCE[j], 
                 THRESHOLD[j], 
                 T_video_import, 
                 T_median_image, 
                 np.mean(T_RS), 
                 np.mean(T_GS), 
                 np.mean(T_positions3D), 
                 T_total,
                 os.path.split(PATH),
                 os.path.split(EXPORT_PATH)[1]]
    
    if EXPORT == 'n':
        TIME_LIST[-1] = []
        
    PARAMETER_LIST.append(TIME_LIST)
    
        
    POSITIONS = pd.DataFrame(columns=['Y', 'X', 'Z', 'I_FS', 'I_GS', 'FRAME'])
    for i in range(np.shape(LOCS)[0]):
        XYZ, I_FS, I_GS, FRAME = LOCS[i, 0], LOCS[i, 1], LOCS[i, 2], i*np.ones_like(LOCS[i, 2])
        DATA = np.concatenate((XYZ, np.expand_dims(I_FS, axis=1), np.expand_dims(I_GS, axis=1), np.expand_dims(FRAME, axis=1)), axis=1)
        POSITIONS = POSITIONS.append(pd.DataFrame(DATA, columns=['Y', 'X', 'Z', 'I_FS', 'I_GS', 'FRAME']))
        
    if EXPORT == 'y':
        # POSITIONS.to_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv', header=True)
        # EXPORT_PATH = PATH[:-4]+'_'+np.str(NUM_FRAMES)+'_FRAMES_RS_TH'+np.str(THRESHOLD).replace('.','')+'_MPD'+np.str(PEAK_MIN_DISTANCE)+'.csv'
        POSITIONS.to_csv(EXPORT_PATH)  # For leptospira data
        print('Results exported to: \n', EXPORT_PATH)
    
    
#%%
# Write runtime to csv file
import csv
    
# counter = len(PARAMETER_LIST)
with open(r'/media/erick/NuevoVol/LINUX_LAP/PhD/Runtime.csv', 'a') as f:
    writer = csv.writer(f)
    for x in range(len(PARAMETER_LIST)):
        writer.writerow(PARAMETER_LIST[x])
    
#%%
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(np.arange(len(T)), np.array(T)/60, '.-'); ax[0].grid()
# ax[0].set_title('Computation time'); ax[0].set_xlabel('Number of frames'); ax[0].set_ylabel('Time (min)')

# ax[1].plot(np.arange(1, len(T)), np.diff(T), '.-'); ax[1].grid()
# ax[1].set_title('Computation time per frame'); ax[1].set_xlabel('Frame number'); ax[1].set_ylabel('Time (s)')
# fig.show()

# POSITIONS = pd.DataFrame(columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME'])
# for i in range(np.shape(LOCS)[0]):
#     XYZ, I_FS, I_GS, FRAME = LOCS[i, 0], LOCS[i, 1], LOCS[i, 2], i*np.ones_like(LOCS[i, 2])
#     DATA = np.concatenate((XYZ, np.expand_dims(I_FS, axis=1), np.expand_dims(I_GS, axis=1), np.expand_dims(FRAME, axis=1)), axis=1)
#     POSITIONS = POSITIONS.append(pd.DataFrame(DATA, columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME']))

#%% Save to file    

if OPTIONS[2] == 'y':
    # POSITIONS.to_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv', header=True)
    EXPORT_PATH = PATH[:-4]+'_'+np.str(NUM_FRAMES)+'_FRAMES_RS_TH'+np.str(THRESHOLD).replace('.','')+'_MPD'+np.str(PEAK_MIN_DISTANCE)+'.csv'
    POSITIONS.to_csv(EXPORT_PATH)  # For leptospira data
    print('Results exported to: \n', EXPORT_PATH)

#%% Plot with f.plot3D
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot

# fig  = pyplot.figure()
# ax = Axes3D(fig)
# for i in range(NUM_FRAMES):
#     f.plot3D(LOCS[i, 0],'Cells in 3D', fig, ax); pyplot.show()
#     pyplot.pause(0.1)
    
#%% Plot with plotly.express
# import plotly.express as px
# import pandas as pd
# import easygui as gui
# from plotly.offline import plot

# # #To plot csv files
# # PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/')
# # POSITIONS = pd.read_csv(PATH, index_col=0)
# fig = px.scatter_3d(POSITIONS, x='X', y='Y', z='Z', color='FRAME')
# fig.update_traces(mode='markers', marker=dict(size=1))
# plot(fig)

#%% Plot with plotly.graph_objects
import pandas as pd
import easygui as gui
import plotly.graph_objects as go
from plotly.offline import plot, iplot

# # To plot csv files
# PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/', filetypes='.csv')
# POSITIONS = pd.read_csv(PATH, index_col=0)
# POSITIONS = pd.read_csv(PATH)

fig = go.Figure(data=[go.Scatter3d(
    x=POSITIONS['X'], 
    y=POSITIONS['Y'], 
    z=POSITIONS['Z'],
    mode='markers', 
    marker=dict(
        size=1,
        color=POSITIONS['FRAME'].values,
        colorscale='Viridis'
        ),
    hovertext=['X+Y+Z+FRAME'],
    hoverinfo='all'
    
)])

fig['layout']['scene']['aspectmode'] =  'cube'
fig.show()
plot(fig)

#iplot(fig)
fig.write_html(PATH[:-3]+'html')


#%% Matplotlib scatter plot
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
# %matplotlib auto

# PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/', filetypes='.csv')
# POSITIONS = pd.read_csv(PATH)

# PKS = A.__array__()
PKS = POSITIONS.__array__()
# np.savetxt('locs.txt', PKS)
fig = pyplot.figure()
ax = Axes3D(fig)

# p = ax.scatter(PKS[:, 0], PKS[:, 1], PKS[:, 2], s=25, marker='o')
p = ax.scatter(POSITIONS['X'], POSITIONS['Y'], POSITIONS['Z'], s=2, marker='o', c=POSITIONS['FRAME'])

ax.tick_params(axis='both', labelsize=10)
ax.set_title('bort6')
fig.colorbar(p)
pyplot.show()

# %%
