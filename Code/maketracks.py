# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

from IPython import get_ipython

# %%
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
DF = pd.read_csv(PATH, index_col=0)

# DF = pd.read_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_rayleighSommerfeld_Results.csv', index_col=0)
# DF= pd.read_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv', index_col=0)
# DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1_500_FRAMES_MODIFIED.csv', index_col=0)
# DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_2001_FRAMES_MODIFIED.csv', index_col=0)
# DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_2001_FRAMES_RS_TH019.csv', index_col=0)

DF.index = np.arange(len(DF))
# D = DF[['X', 'Y', 'Z', 'FRAME']].values
D = DF.values
# DD = A[:, :3]



# # Detect tracks using ML

# %%
# KMeans, DBSCAN or HA
# method = 'DBSCAN'
method = gui.choicebox(msg='Choose a tracking Algorithm', title='Choose', choices=['KMeans', 'DBSCAN', 'HA'])


if method == 'KMeans':

    #% K-Means for track detection
    T0_kmeans = time.time()
    kmeans = cl.KMeans(n_clusters=D[D[:, 5] == 0].shape[0], init='k-means++').fit(DF[['X', 'Y', 'Z']])
    DF['PARTICLE'] = kmeans.labels_
    LINKED = DF
    T_kmeans = time.time() -  T0_kmeans

elif method == 'DBSCAN':

    #% Density Based Spatial Clustering (DBSC)
    T0_DBSCAN = time.time()
    DBSCAN = cl.DBSCAN(eps=5, min_samples=8).fit(DF[['X', 'Y', 'Z']])
    DF['PARTICLE'] = DBSCAN.labels_
    LINKED = DF
    L = LINKED.drop(np.where(LINKED.PARTICLE.values == -1)[0])
    T_DBSCAN = time.time() - T0_DBSCAN

elif method == 'HA':

    #% Hungrian algorithm
    T0_HA = time.time()
    FRAME_DATA = np.empty(int(DF['FRAME'].max()), dtype=object)
    SIZE = np.empty(FRAME_DATA.shape[0])
    for i in range(FRAME_DATA.shape[0]):
        FRAME_DATA[i] = D[D[:, 5] == i]
        SIZE[i] = FRAME_DATA[i].shape[0]
        
    # To make a square cost matrix
    # SIZES = np.empty(FRAME_DATA.shape[0])
    # for i in range(FRAME_DATA.shape[0]):
    #     if SIZE.max() != FRAME_DATA[i].shape[0]:
    #         DIFF = int(SIZE.max() - FRAME_DATA[i].shape[0])
    #         FRAME_DATA[i] = np.pad(FRAME_DATA[i], ((0, DIFF), (0, 0)), 'constant', constant_values=0)
    #     SIZES[i] = FRAME_DATA[i].shape[0]
            
    
    # particle_index = np.empty((int(SIZES[0]), len(FRAME_DATA)), dtype='int')
    TRACKS = [FRAME_DATA[0]]
    sizes = np.empty(len(FRAME_DATA), dtype='int')
    sizes[0] = len(FRAME_DATA[0])
    for k in range(len(FRAME_DATA)-1):  
        print(k)
        
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
    LINKED = TR
    T_HA = time.time() - T0_HA
    print(T_HA)

LINKED = LINKED[LINKED.PARTICLE != -1]
# ID = np.where(SHAPES >= 1000)[0]
# IND  = np.in1d(TR.PARTICLE.values,  ID, invert=True)
# LINKED = TR.loc[IND, :]
    


# # Cubic Spline Smoothing

# %%
# LINKED = LINKED[LINKED.PARTICLE != -1]

# For Archea data that looks messy
# value_counts = LINKED.PARTICLE.value_counts()
# values = value_counts[value_counts.values < 1000]
# idx = LINKED.PARTICLE.isin(values.index)
# LINKED = LINKED.iloc[idx.values]

spline_degree = 3  # 3 for cubic spline
particle_num = np.sort(LINKED.PARTICLE.unique())
T0_smooth = time.time()
smoothed_curves = -np.ones((1, 4))
for pn in particle_num:
    # Do not use this
    # L = LINKED[LINKED.PARTICLE == pn].values
    # X = f.smooth_curve(L, spline_degree=spline_degree, lim=20, sc=3000)
    
    L = LINKED[LINKED.PARTICLE == pn]
    if len(L) < 100:
        continue
    X = f.csaps_smoothing(L, smoothing_condition=0.999999, smooth_data=True)
    
    if X != -1:
        smoothed_curves = np.vstack((smoothed_curves, np.stack((X[0], X[1], X[2], pn*np.ones_like(X[1])), axis=1))) 

smoothed_curves = smoothed_curves[1:, :]
smoothed_curves_df = pd.DataFrame(smoothed_curves, columns=['X', 'Y' ,'Z', 'PARTICLE'])
T_smooth = time.time() - T0_smooth

# smoothed_curves_df.to_csv(PATH[:-4]+'_DBSCAN_smooth_200.csv', index=False)


# # RDP algorithm for each dimension separately

# %%
#%% RDP algorithm for each dimension separately.
get_ipython().run_line_magic('matplotlib', 'qt')
from rdp import rdp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# L = LINKED[LINKED.PARTICLE == 6]
L = smoothed_curves_df[smoothed_curves_df.PARTICLE == 67]
xx, yy, zz = L.X.values, L.Y.values, L.Z.values
xx = xx.reshape(len(xx), 1)
yy = yy.reshape(len(yy), 1)
zz = zz.reshape(len(zz), 1)

t = np.linspace(0, 1, len(xx))
t = t.reshape(len(t), 1)

stack =  np.hstack((xx, yy, zz))

eps = 0.05
# eps3d = 0.1
# xyz = rdp(stack, epsilon=eps3d)
x = rdp(np.hstack((t, xx)), epsilon=eps)
y = rdp(np.hstack((t, yy)), epsilon=eps)
z = rdp(np.hstack((t, zz)), epsilon=eps)


# %%
XYZ = np.sort(np.concatenate((x[:, 0], y[:, 0], z[:, 0]), axis=0))
unique = np.unique(XYZ, axis=0)
unique_id, _ = np.where(t == unique)
turns = np.concatenate((xx[unique_id], yy[unique_id], zz[unique_id]), axis=1)

# fig = plt.figure(figsize=(7, 4.5))
# ax0 = plt.subplot2grid((6,6), (0, 0), 2, 2)
# ax0.plot(t, xx, 'b-')
# ax0.plot(x[:, 0], x[:, 1], 'r.')
# ax0.set_ylabel('x')
# ax0.set_title('eps = '+np.str(eps))
# ax0.grid()

# ax1 = plt.subplot2grid((6, 6), (2, 0), 2, 2)
# ax1.plot(t, yy, 'b-')
# ax1.plot(y[:, 0], y[:, 1], 'r.')
# ax1.set_ylabel('y')
# ax1.grid()

# ax2 = plt.subplot2grid((6, 6), (4, 0), 2, 2)
# ax2.plot(t, zz, 'b-')
# ax2.plot(z[:, 0], z[:, 1], 'r.')
# ax2.set_ylabel('z')
# ax2.grid()

# #fig = plt.figure(figsize=(14, 9))
# # ax3 = fig.add_subplot(111, projection='3d')
# ax3 = plt.subplot2grid((6, 6), (0, 2), 6, 6, projection='3d')
# ax3.plot(L.X, L.Y, L.Z, 'b-', label='original data')
# ax3.plot(turns[:, 0], turns[:, 1], turns[:, 2], 'r.', label='Turns')
# ax3.set_title('Merged Turns')


# %%
# Bokeh plots
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook, push_notebook
from bokeh.layouts import row, column, gridplot
# output_notebook()


TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select,hover"
p1 = figure(title='eps = '+np.str(eps), y_axis_label='x', 
            tools=TOOLS)
p1.line(t[:, 0], xx[:, 0], legend_label="Temp.", line_width=2)
# p1.circle(x[:, 0], x[:, 1], fill_color='red', line_color='red', radius=0.005)

p2 = figure(title='eps = '+np.str(eps), y_axis_label='y', 
            tools=TOOLS)
p2.line(t[:, 0], yy[:, 0], legend_label="Temp.", line_width=2)
# p2.circle(y[:, 0], y[:, 1], fill_color='red', line_color='red', radius=0.005)

p3 = figure(title='eps = '+np.str(eps), y_axis_label='z', 
            tools=TOOLS)
p3.line(t[:, 0], zz[:, 0], legend_label="Temp.", line_width=2)
# p3.circle(z[:, 0], z[:, 1], fill_color='red', line_color='red', radius=0.005)

grid = gridplot([p1, p2, p3], ncols=1, plot_width=900, plot_height=150)
show(grid)

fig = plt.figure(figsize=(9, 4.5))
ax3 = fig.add_subplot(111, projection='3d')
# ax3 = plt.subplot2grid((6, 6), (0, 0), 6, 6, projection='3d')
ax3.plot(L.X, L.Y, L.Z, 'b-', label='original data')
ax3.plot(turns[:, 0], turns[:, 1], turns[:, 2], 'r.', label='Turns')
ax3.set_title('Merged Turns')

# # RDP algorithm for 3D trajectory

# %%
#%% RDP algorithm for 3D trajectoy simulataneously
eps3d = 0.5
xyz = rdp(stack, epsilon=eps3d)

fig = plt.figure(figsize=(9, 4.5))
ax0 = plt.subplot2grid((6, 6), (0, 0), 6, 3, projection='3d')
ax0.set_facecolor('none')
ax0.plot(L.X, L.Y, L.Z, 'b-', label='original data')
ax0.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'r.', label='Turns')
ax0.set_title('Smoothed track')

ax1 = plt.subplot2grid((6, 6), (0, 3), 6, 3, projection='3d')
ax1.set_facecolor('none')
ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'b-', label='RDP reconstructed eps = '+np.str(eps3d))
ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'r.', label='Turns')
ax1.set_title('RDP Simplification')
ax1.legend(loc='best')


# %%
# Use derivatives of r (distance to origin) to sense changes in direction
from scipy.signal import find_peaks
r = np.sqrt(L.X.values**2 + L.Y.values**2 + L.Z.values**2)
diff = np.diff(r)
diff2 = np.diff(diff)
peaks, _ = find_peaks(diff, height=0.04)
peaks2, _ = find_peaks(diff2, height=np.min(diff2)-1)   # to get all local maxima

fig = plt.figure(figsize=(9, 4.5))
ax = fig.add_subplot(311)
ax.plot(r)

axx = fig.add_subplot(312)
axx.plot(np.arange(len(diff)), diff)
axx.plot(peaks, diff[peaks], '.')
axx = fig.add_subplot(313)
axx.plot(np.arange(len(diff2)), diff2)
axx.plot(peaks2, diff2[peaks2], '.')


# %%
fig = plt.figure(figsize=(9, 7))
axxx = fig.add_subplot(121, projection='3d')
axxx.plot(L.X, L.Y, L.Z, '-', label='original data')
axxx.plot(L.X.values[peaks], L.Y.values[peaks], L.Z.values[peaks], 'r.')

axxx = fig.add_subplot(122, projection='3d')
axxx.plot(L.X, L.Y, L.Z, '-', label='original data')
axxx.plot(L.X.values[peaks2], L.Y.values[peaks2], L.Z.values[peaks2], 'r.')

plt.show()


# %%
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
#%matplotlib qt

p_number = 9
CURVE_1 = LINKED[LINKED.PARTICLE == p_number]
CURVE_2 = smoothed_curves_df[smoothed_curves_df.PARTICLE == p_number]

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(CURVE_1.X, CURVE_1.Y, CURVE_1.Z, 'r.', label='Detected Positions', c=np.arange(len(CURVE_1.X)))
ax.plot(CURVE_2.X, CURVE_2.Y, CURVE_2.Z, 'r-', label='Smoothed Curve')
pyplot.show()


# %%
import plotly.express as px
import pandas as pd
import plotly
from plotly.offline import plot, iplot
plotly.offline.init_notebook_mode()


# For Archea data that looks messy
# value_counts = LINKED.PARTICLE.value_counts()
# values = value_counts[value_counts.values < 2000]
# idx = LINKED.PARTICLE.isin(values.index)
# CURVE = LINKED.iloc[idx.values]

# CURVE = LINKED[LINKED.PARTICLE == 4]
# CURVE = LINKED[(LINKED.PARTICLE != -1)]
# CURVE = pd.DataFrame(smoothed_curves, columns=['X', 'Y', 'Z', 'PARTICLE'])
# CURVE = CURVE[CURVE.PARTICLE != -1]
# CURVE = LINKED[LINKED.PARTICLE == 28]
CURVE = smoothed_curves_df[smoothed_curves_df.PARTICLE != -1]
# CURVE = smoothed_curves_df[smoothed_curves_df.PARTICLE == 28]

# fig = px.scatter_3d(CURVE, x='X', y='Y', z='Z', color='PARTICLE')
fig = px.line_3d(CURVE, x='X', y='Y', z='Z', color='PARTICLE')
fig.update_traces(marker=dict(size=1))
plot(fig)
# iplot(fig)

# fig.write_html(PATH[:-3]+'html')
# fig.write_html(PATH[:-4]+'_HA.html')



# %%
import plotly.graph_objects as go
import plotly
from plotly.offline import iplot
plotly.offline.init_notebook_mode()

x = CURVE['X']
y = CURVE['Y']
z = CURVE['Z']
particle = CURVE['PARTICLE']

trace = []
for i in pd.unique(CURVE.PARTICLE):
    dat = CURVE[CURVE['PARTICLE'] == i]
    x = dat['X']
    y = dat['Y']
    z = dat['Z']
    particle = dat['PARTICLE']

    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers+lines',
        marker=dict(
            size=1,
            color=i,
            colorscale='Viridis',
            opacity=0.8),
        hoverinfo='all',
        )
    
    trace.append(trace1)


data = trace
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0))

fig = go.Figure(data=data, layout=layout)
plot(fig)
# iplot(fig, filename='3d-pubg-plot')


# %%



