#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:17:58 2021

@author: erick
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import easygui as gui
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

path = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/')

data = pd.read_csv(path, index_col=0)

#%%
# p_number = 2
# CURVE_1 = LINKED[LINKED.PARTICLE == p_number]
# CURVE_2 = smoothed_curves_df[smoothed_curves_df.PARTICLE == p_number]
# CURVE_2 = smoothed_curves_df
particle = 0
particle_data = data[data['PARTICLE'] == particle]

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data.X, data.Y, data.Z, 'r.', label='Detected Positions', c=np.arange(len(data.X)))
ax.scatter(particle_data.X, particle_data.Y, particle_data.Z, 'r.', label='Detected Positions', c=np.arange(len(particle_data.X)))
# ax.plot(CURVE_2.X, CURVE_2.Y, CURVE_2.Z, 'r-', label='Smoothed Curve')
pyplot.show()