#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 21:08:38 2022

@author: erick
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import easygui as gui
import functions as f
from tqdm import tqdm
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

path = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/Thesis/Results/1/')
smoothed_curves_df = pd.read_csv(path)

#%%
# For Noramal video
particle_num = [7.1, 4.1, 26.0, 3.1, 16.1]   # Ecoli DHM 0.99, 0.95, 0.9, 0.85



# Fov BS video
# particle_num = np.unique(smoothed_curves_df['PARTICLE'])   # Coli Optical LUT80 0.99, Speed
# particle_num = [6.1, 1.1, 5.1, 19.1] # Ecoli DHM 0.95, 0.99 Speed 20.62
# particle_num = [8.0, 2.0, 5.0, 1.0, 4.0]   # Ecoli GPU LUT80 0.99
# particle_num = [8.0, 3.0, 5.0, 1.1, 4.0]     # Ecoli GPU LUT160 0.99
# particle_num = [12.0, 10.0, 8.1, 1.1, 6.0]
# particle_num = [8.0, 10.0, 3.1, 2.2, 7.1]       # Ecoli Optical LUT80 0.95, 0.9, 

xx, yy, zz, tt, pp, sp = -1, -1, -1, -1, -1, -1

for pn in particle_num:
    s = smoothed_curves_df[smoothed_curves_df['PARTICLE'] == pn]
    print(pn, len(s))

    if len(s) > 100:
        speed, x, y, z, t = f.get_speed(s)
        xx = np.hstack((xx, x))
        yy = np.hstack((yy, y))
        zz = np.hstack((zz, z))
        tt = np.hstack((tt, t))
        pp = np.hstack((pp, pn*np.ones(len(t))))
        sp = np.hstack((sp, speed))
    
# expath = '/media/erick/NuevoVol/LINUX_LAP/PhD/Thesis/Results/1/Archea/Plots/'
tracks_w_speed = pd.DataFrame(np.transpose([xx[1:], yy[1:], zz[1:], tt[1:], pp[1:], sp[1:]]), columns=['X', 'Y', 'Z', 'TIME', 'PARTICLE', 'SPEED'])
tracks_w_speed.to_csv(path[:-4]+'_speed.csv')


# PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/', filetypes='.csv')
# tracks_w_speed = pd.read_csv(PATH, index_col=False)

fig = plt.figure(4, dpi=150)
ax = fig.add_subplot(111, projection='3d')

# p = ax.scatter(tracks_w_speed['Y'], tracks_w_speed['X'], tracks_w_speed['Z'], c=tracks_w_speed['SPEED'], marker='.', s=20)
# cbar = plt.colorbar(p)
# cbar.set_label('Speed ($\mu ms^{-1}$)')

for pn in particle_num:
    # s = smoothed_curves_df[smoothed_curves_df['PARTICLE'] == pn]
    s = tracks_w_speed[tracks_w_speed['PARTICLE'] == pn]
    ax.plot(s['X'], s['Y'], s['Z'], linewidth=2)
    # ax.scatter(s['Y'], s['X'], s['Z'])

ax.axis('tight')
ax.set_title('$\it{Escherichia \ Coli}$', fontsize=40)  # $\it{Escherichia \ Coli}$
ax.set_xlabel('y ($\mu$m)', fontsize=20)
ax.set_ylabel('x ($\mu$m)', fontsize=20)
ax.set_zlabel('-z ($\mu$m)', fontsize=20)
# ax.set_zlim(bottom=0, top=40)

plt.figure(3)
plt.hist(tracks_w_speed['SPEED'], 100)
mean_speed = tracks_w_speed['SPEED'].mean()
print(mean_speed)
plt.title('Speed: $\mu$ = ' + str(np.float16(mean_speed)) + ' $\mu m s^{-1}$', fontsize=40)
plt.xlabel('Speed ($\mu m s^{-1}$)', fontsize=20)
plt.ylabel('Frequency', fontsize=20)

pyplot.show()

#%%
import plotly.express as px
import pandas as pd
import easygui as gui
from plotly.offline import plot

pns = tracks_w_speed.PARTICLE.unique()
curve = tracks_w_speed[tracks_w_speed['PARTICLE'] == pns[1]]

fig = px.line_3d(tracks_w_speed, x='X', y='Y', z='Z', color='PARTICLE', hover_data=['TIME'])
# fig = px.scatter_3d(tracks_w_speed, x='X', y='Y', z='Z', color='SPEED')
fig.update_traces(marker=dict(size=1))

plot(fig)

#%% Re orientation Angle with RDP

def get_angles_rdp(sd, epsilon, angle_threshold):
    """ Re-orientation event angle"""
    
    from rdp import rdp
    from scipy.signal import find_peaks
    
    epsilon = 0.5
    
    x = sd[['X', 'Y', 'Z']].values
    t = sd['TIME'].values
    p = sd['PARTICLE'].values[0]
    
    coords = rdp(x, epsilon, return_mask=True)   # RDP reconstruction
    xyz = np.hstack((x[coords, :], p*np.ones((np.count_nonzero(coords), 1))))
    xyz = np.hstack((xyz, np.expand_dims(np.where(coords==True)[0], axis=1)))
    
    
    cos = np.ones(len(xyz))
    Dr = np.ones(len(xyz))
    for i in range(1, len(xyz)-1):
        a1 = xyz[i-1, :3]
        a2 = xyz[i, :3]
        a3 = xyz[i+1, :3]
        d21 = a2 - a1
        d32 = a3 - a2
        cos[i] = np.dot(d21, d32) / (np.sqrt(np.sum(d21**2)) * np.sqrt(np.sum(d32**2)))
        
    angles = np.arccos(cos) * 180 / np.pi
    
    
    pks, _ = find_peaks(angles, height=angle_threshold)
    ids = xyz[pks, -1].astype('int')                          # indices of original array
    
    
    
    # plt.figure()
    # plt.plot(t[coords], angles)
    # plt.plot(t[ids], angles[pks], 'ro')
    
    # plt.figure()
    # plt.hist(angles, 30)
    
    # plt.figure(figsize=(9, 4.5))
    # ax0 = plt.subplot2grid((6, 6), (0, 0), 6, 3, projection='3d')
    # ax0.set_facecolor('none')
    # ax0.plot(sd.X, sd.Y, sd.Z, 'b-', label='original data')
    # ax0.plot(sd.X[ids], sd.Y[ids], sd.Z[ids], 'ro', label='original data')
    # ax0.set_title('Smoothed track')

    # ax1 = plt.subplot2grid((6, 6), (0, 3), 6, 3, projection='3d')
    # ax1.set_facecolor('none')
    # ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'b-', label='RDP reconstructed eps = '+np.str(epsilon))
    # ax1.plot(sd.X[ids], sd.Y[ids], sd.Z[ids], 'ro', label='Turns')
    # ax1.set_title('RDP Simplification')
    # ax1.legend(loc='best')
    
    # angles = np.hstack((angles.reshape((len(angles), 1)), p*np.ones((len(angles), 1))))
    
    return ids, xyz, angles       # index of turns of original array, RDP simplification, turn angles



eps = 0.5
pnum = tracks_w_speed['PARTICLE'].unique()

indices = []
angles = []
recs = []
for pn in tqdm(pnum):
    sd = tracks_w_speed[tracks_w_speed['PARTICLE'] == pn]
    i, r, a = get_angles_rdp(sd, eps, 45)
    indices.append(i)
    angles.append(a)
    recs.append(r)
    
angles = np.concatenate(angles)
recs = np.concatenate(recs)

recs= pd.DataFrame(recs, columns=['X', 'Y', 'Z', 'PARTICLE', 'INDEX_ORIGIN'])
amean = angles.mean()

#%%
m = plt.hist(angles, 32)[0]
plt.vlines(amean, ymin=0, ymax=m.max(), colors='red')
plt.title('Mean: '+ str(round(amean, 3)))
plt.grid()

#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot


fig = pyplot.figure()
ax = Axes3D(fig)

pn = 5

sd =  tracks_w_speed[tracks_w_speed['PARTICLE'] == pn]
rec = recs[recs['PARTICLE'] == pn]

ax.plot(sd.X, sd.Y, sd.Z, linewidth=3)
ax.scatter(rec.X, rec.Y, rec.Z, s=50, marker='o', c='red')


ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (um)', fontsize='18')
ax.set_ylabel('y (um)', fontsize='18')
ax.set_zlabel('z (um)', fontsize='18')
# fig.colorbar(p, ax=ax)
pyplot.show()



#%% With LW metric
def xi_metric(sd, threshold):
    
    from scipy.signal import find_peaks
    
    x = sd[['X', 'Y', 'Z']].values
    v = sd['SPEED'].values
    t = sd['TIME'].values
    
    
    vmean = v.mean()
    dt = np.diff(t)
    
    tau = 1
    xi = np.zeros((len(x)))
    dot = np.zeros((len(x)))
    cos = np.zeros((len(x)))
    arcdot = np.zeros(len(x))
    dot = []
    
    for i in range(tau, len(x)-tau):
        
        dd = tau
        a1 = x[i-dd, :]
        a2 = x[i, :]
        a3 = x[i+dd, :]
        d21 = a2 - a1
        d32 = a3 - a2 
        dot.append(np.dot(d21, d32))
        
        arcdot[i] = np.arccos(np.dot(d21, d32) / (np.sqrt(np.sum(d21**2)) * np.sqrt(np.sum(d32**2)))) * 180 / np.pi
        
        cos[i] = np.dot(d21, d32) / (np.sqrt(np.sum(d21**2)) * np.sqrt(np.sum(d32**2)))
        const = 1 - v[i]/vmean
        # psi[i] = const * np.abs(np.dot(d21, d32)) / (t[i+1] - t[i-1])
        
        xi[i] = const * arcdot[i] / (t[i+1] - t[i-1])
    
    angles = np.arccos(cos) * 180 / np.pi    
    angles[angles > 45] = 0
    
    pks, _ = find_peaks(xi, height=threshold)

    # plt.plot(t, xi)
    # plt.plot(t[pks], xi[pks], 'ro')
    

    return pks, xi, angles


# Test
pn = 9
sd = tracks_w_speed[tracks_w_speed['PARTICLE'] == pn]

pks, xi, angles = xi_metric(sd, 40)

plt.figure()
plt.hist(angles)


# Xi plot
plt.figure()
plt.plot(sd['TIME'], xi)
plt.plot(sd['TIME'].values[pks], xi[pks], 'ro')
# plt.xlabel('Time (s)', fontsize=10)
plt.ylabel('$\Xi$ (degrees $s^{-1}$)', fontsize=10)
plt.title('Archea Track', fontsize=10)
plt.grid()


# Track with event detected
fig = pyplot.figure()
ax = Axes3D(fig)


ax.plot(sd.X, sd.Y, sd.Z, linewidth=3)
ax.scatter(sd.X.values[pks[-1]], sd.Y.values[pks[-1]], sd.Z.values[pks[-1]], s=50, marker='o', c='red')


ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (um)', fontsize='18')
ax.set_ylabel('y (um)', fontsize='18')
ax.set_zlabel('z (um)', fontsize='18')
# fig.colorbar(p, ax=ax)
pyplot.show()

#%%
X = sd[['X', 'Y', 'Z']].values
peak = pks[-1]

theta = []


for d in range(1, 200):
    a21 = X[peak, :] - X[peak-d, :]
    a32 = X[peak+d, :] - X[peak, :]
    
    norm21 = np.sqrt(np.dot(a21, a21))
    norm32 = np.sqrt(np.dot(a32, a32))
    
    th = np.arccos(np.dot(a21, a32) / (norm21*norm32)) * 180 / np.pi
    # print(theta)
    theta.append(th)

plt.plot(range(1, 200), theta)



#%%
#######################################################################################################
###
###                                     Test with LW track
###
#######################################################################################################

file = '/media/erick/NuevoVol/LINUX_LAP/PhD/Thesis/Results/1/Archea/LW track with reversals/track243_1_0.txt'
sd_lw = np.loadtxt(file)
sd_lw = pd.DataFrame(sd_lw[:,:4], columns=['TIME', 'X' ,'Y', 'Z'])
sd_lw['FRAME'] = np.arange(len(sd_lw))

sd_lw_smooth = np.array(f.csaps_smoothing(sd_lw, 0.5, False, 6)).transpose()
sd_lw = pd.DataFrame(sd_lw_smooth, columns=['X', 'Y', 'Z', 'TIME'])

sd_lw_speed = np.array(f.get_speed(sd_lw)).transpose()
sd_lw_speed = pd.DataFrame(sd_lw_speed, columns=['SPEED', 'X', 'Y', 'Z', 'TIME'])


# LW metric
pks, xi, angles = xi_metric(sd_lw_speed, 160)

plt.figure()
plt.subplot(2,1,1)
plt.plot(sd_lw_speed['TIME'].values, xi)
plt.plot(sd_lw_speed['TIME'].values[pks], xi[pks], 'ro')
# plt.xlabel('Time (s)', fontsize=10)
plt.ylabel('$\Xi$ ($\mu m^{2}s^{-1}$)', fontsize=10)
plt.title('LW Track', fontsize=10)
plt.grid()

plt.subplot(2,1,2)
plt.plot(sd_lw_speed['TIME'].values, angles)
plt.plot(sd_lw_speed['TIME'].values[pks], angles[pks], 'ro')
plt.xlabel('Time (s)', fontsize=10)
plt.ylabel('Angles (degrees)', fontsize=10)
# plt.title('Angles', fontsize=10)
plt.grid()

##
fig = plt.figure()
ax = Axes3D(fig)

ax.plot(sd_lw_speed.X, sd_lw_speed.Y, sd_lw_speed.Z, linewidth=3)
ax.scatter(sd_lw_speed.X.values[pks], sd_lw_speed.Y.values[pks], sd_lw_speed.Z.values[pks], s=50, marker='o', c='red')


ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (um)', fontsize='18')
ax.set_ylabel('y (um)', fontsize='18')
ax.set_zlabel('z (um)', fontsize='18')
# fig.colorbar(p, ax=ax)
pyplot.show()


#%% RPD
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

sd_lw_speed['PARTICLE'] = np.ones(len(sd_lw_speed))

epsilon = 0.5
angle_threshold = 45
ids, rec, angles = get_angles_rdp(sd_lw_speed, epsilon, angle_threshold)
 

plt.figure(figsize=(9, 4.5))
ax0 = plt.subplot2grid((6, 6), (0, 0), 6, 3, projection='3d')
ax0.set_facecolor('none')
ax0.plot(sd_lw_speed.X, sd_lw_speed.Y, sd_lw_speed.Z, 'b-', label='original data')
ax0.plot(sd_lw_speed.X[ids], sd_lw_speed.Y[ids], sd_lw_speed.Z[ids], 'ro', label='original data')
ax0.set_title('Smoothed track')

ax1 = plt.subplot2grid((6, 6), (0, 3), 6, 3, projection='3d')
ax1.set_facecolor('none')
ax1.plot(rec[:, 0], rec[:, 1], rec[:, 2], 'b-', label='RDP reconstructed eps = '+np.str(epsilon))
ax1.plot(sd_lw_speed.X.values[ids], sd_lw_speed.Y.values[ids], sd_lw_speed.Z.values[ids], 'ro', label='Turns')
ax1.set_title('RDP Simplification')
ax1.legend(loc='best')


#%%
def plotly_scatter(tracks):
    import plotly.express as px
    import plotly
    from plotly.offline import plot, iplot
    
    fig = px.scatter_3d(tracks, x='X', y='Y', z='Z', color='TIME', labels='TIME')
    fig.update_traces(marker=dict(size=1))
    plot(fig)
    # iplot(fig)

plotly_scatter(sd)
