#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 16:13:58 2021

@author: erick
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import easygui as gui
import os


# path = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/')
path = gui.diropenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/')
path_splitted = os.path.split(path)
file_list = os.listdir(path)

#%%
frame = np.arange(len(file_list))

# data = zip(frame, file_list)

data = -1*np.ones((1, 7))
for frame, file in zip(frame, file_list):
    frame_data = np.loadtxt(path+'/'+file)
    frame_data = frame_data[:-1, :]
    f = np.ones((len(frame_data), 1))
    frame_data_stack = np.hstack((frame_data, f, f , frame*np.ones((len(frame_data), 1))))
    data = np.vstack((data, frame_data_stack))
    print(frame, file)

data = data[1:, 1:]
# data = data[:, 1:]

#%%
data_table = pd.DataFrame(data, columns=('X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME'))

data_table.to_csv(path_splitted[:-1][0]+'/40x_Bdello_frames.csv')

