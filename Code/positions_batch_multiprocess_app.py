#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 1 22:04:09 2022

@author: erick
"""

""" Positions Batch App"""
#%%
#from email.policy import default
import streamlit as st
import os
import numpy as np
import PySimpleGUI as sg
import pandas as pd
import matplotlib.pyplot as plt
import functions as f
from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import tqdm

#########################################
#          Import Data
#########################################

st.header("Import Video File")

st.cache()
def vid_import():
    layout = [
        [sg.Text('Select AVi File recording', 
                    size=(35, 1)), 
                    # sg.In(default_text='Select a video'), 
                    sg.FileBrowse(initial_folder='/media/erick/NuevoVol/LINUX_LAP/PhD/Test/Ecoli/07/')],
         [sg.Button('Read'), sg.Cancel()]
    ]


    select = st.button("Select AVI File")

    if select:
        window = sg.Window('Holography video input', layout)
        
        while True:
            event, value = window.Read()
            values = list(value.values())
            if event == "Cancel":
                break
            if event == "Read":
                break
        window.Close()
        
    # st.write(values[0])

    return f.videoImport(values[0], 0)

# VID = vid_import()

with st.form("Parameters"):
    VID = vid_import()
    
    N = st.number_input('Index of Refraction', value=1.3226)
    LAMBDA = st.number_input('Wavelength (um)', value=0.642)
    MPP = st.number_input('Magnification', value=20)
    SZ = st.number_input('Step Size (um)', value=5)
    NUMSTEPS = st.number_input('Number of Steps', value=150)
    THRESHOLD = st.number_input('Threshold', value=0.1)
    PMD = st.number_input('PEak Min Distance', value=20)
    FRAME_RATE = st.number_input('Frame Rate', value=50)
    INVERT_VIDEO = st.radio('Invert Video?', (False, True))
    export = st.radio('Export?', (True, False))
    num_frames = st.number_input('Number of Frames for Calculation', value=np.shape(VID)[-1])
    # params.append(pd.DataFrame([values[1:]], columns=['N', 'Wavelength', 'MPP', 'SZ', 'Step #', 'Threshold', 'MPD', 'FR','Invert', 'Export', 'Num Frames']))
    # submitted = st.form_submit_button("S")
    
    submitted = st.form_submit_button("do")

    if submitted:
        st.write("Starting")
        ni, nj, nk = np.shape(VID)

        if INVERT_VIDEO:
            for i in range(nk):
                VID[:, :, i] = VID[:, :, i].max() - VID[:, :, i]

        FRAMES_MEDIAN = 20
        I_MEDIAN = f.medianImage(VID, FRAMES_MEDIAN)
        # I_MEDIAN = np.ones((VID.shape[0], VID.shape[1]))

        if num_frames[k] == []:
            NUM_FRAMES = nk
        else:
            NUM_FRAMES = num_frames
            
        IT = np.empty((NUM_FRAMES), dtype=object)
        MED = np.empty((NUM_FRAMES), dtype=object)
        n = np.empty((NUM_FRAMES), dtype=object)
        lam = np.empty((NUM_FRAMES), dtype=object)
        mpp = np.empty((NUM_FRAMES), dtype=object)
        sz = np.empty((NUM_FRAMES), dtype=object)
        numsteps = np.empty((NUM_FRAMES), dtype=object)
        threshold = np.empty((NUM_FRAMES), dtype=object)
        pmd = np.empty((NUM_FRAMES), dtype=object)

        for i in range(NUM_FRAMES):
            IT[i] = VID
            MED[i] = I_MEDIAN
            n[i] = N
            lam[i] = LAMBDA
            mpp[i] = MPP
            sz[i] = SZ
            numsteps[i] = NUMSTEPS
            threshold[i] = THRESHOLD
            pmd[i] = PMD

        pool = Pool(cpu_count())     # Number of cores to use
        results = []
        #     T0 = time.time()

        # print('Processing File '+str(k+1)+' of '+str(len(PATH))+': '+ os.path.split(PATH[k])[-1])
        # print('Parameters: ')
        # print(params[k])
        for _ in tqdm(pool.imap_unordered(f.positions_batch, zip(IT, MED, n, lam, mpp, sz, numsteps, threshold, pmd)), 
                        total=NUM_FRAMES):
            results.append(_)
            
        # for i, item in enumerate(pool.imap_unordered(f.positions_batch, zip(IT, MED, n, lam, mpp, sz, numsteps, threshold, pmd))):
        #     sg.one_line_progress_meter('This is my progress meter!', 
        #                                i+1, 
        #                                NUM_FRAMES,
        #                                'Processing File '+str(k+1)+' of '+str(len(PATH))+': ',
        #                                os.path.split(PATH[k])[-1],
        #                                values[1:],
        #                                orientation='h')
        #     results.append(item)
            
        pool.close()
        pool.join()

        POSITIONS = pd.DataFrame(columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME', 'TIME'])
        for i in range(NUM_FRAMES):

            X = results[i][0][0]
            Y = results[i][1][0]
            Z = results[i][2][0]
            I_FS = results[i][3][0]
            I_GS = results[i][4][0]
            FRAME = i*np.ones_like(results[i][0][0])
            TIME = FRAME / FRAME_RATE[k]
            
            DATA = np.concatenate((np.expand_dims(X, axis=1), 
                                    np.expand_dims(Y, axis=1), 
                                    np.expand_dims(Z, axis=1), 
                                    np.expand_dims(I_FS, axis=1), 
                                    np.expand_dims(I_GS, axis=1), 
                                    np.expand_dims(FRAME, axis=1), 
                                    np.expand_dims(TIME, axis=1)), 
                                    axis=1)
            # POSITIONS = POSITIONS.append(pd.DataFrame(DATA, 
                                                        # columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME', 'TIME']))
            
            POSITIONS = pd.concat([POSITIONS, pd.DataFrame(DATA, 
                                                        columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME', 'TIME'])],
                                    ignore_index=True)

        POSITIONS = POSITIONS.astype('float')
        POSITIONS['TIME'] = POSITIONS['TIME'].round(2)

        st.dataframe(POSITIONS)




# ni, nj, nk = np.shape(VID)

# if INVERT_VIDEO:
#     for i in range(nk):
#         VID[:, :, i] = VID[:, :, i].max() - VID[:, :, i]

# FRAMES_MEDIAN = 20
# I_MEDIAN = f.medianImage(VID, FRAMES_MEDIAN)
# # I_MEDIAN = np.ones((VID.shape[0], VID.shape[1]))

# if num_frames[k] == []:
#     NUM_FRAMES = nk
# else:
#     NUM_FRAMES = num_frames
    
# IT = np.empty((NUM_FRAMES), dtype=object)
# MED = np.empty((NUM_FRAMES), dtype=object)
# n = np.empty((NUM_FRAMES), dtype=object)
# lam = np.empty((NUM_FRAMES), dtype=object)
# mpp = np.empty((NUM_FRAMES), dtype=object)
# sz = np.empty((NUM_FRAMES), dtype=object)
# numsteps = np.empty((NUM_FRAMES), dtype=object)
# threshold = np.empty((NUM_FRAMES), dtype=object)
# pmd = np.empty((NUM_FRAMES), dtype=object)

# for i in range(NUM_FRAMES):
#     IT[i] = VID
#     MED[i] = I_MEDIAN
#     n[i] = N
#     lam[i] = LAMBDA
#     mpp[i] = MPP
#     sz[i] = SZ
#     numsteps[i] = NUMSTEPS
#     threshold[i] = THRESHOLD
#     pmd[i] = PMD

# pool = Pool(cpu_count())     # Number of cores to use
# results = []
# #     T0 = time.time()

# # print('Processing File '+str(k+1)+' of '+str(len(PATH))+': '+ os.path.split(PATH[k])[-1])
# # print('Parameters: ')
# # print(params[k])
# for _ in tqdm(pool.imap_unordered(f.positions_batch, zip(IT, MED, n, lam, mpp, sz, numsteps, threshold, pmd)), 
#                 total=NUM_FRAMES):
#     results.append(_)
    
# # for i, item in enumerate(pool.imap_unordered(f.positions_batch, zip(IT, MED, n, lam, mpp, sz, numsteps, threshold, pmd))):
# #     sg.one_line_progress_meter('This is my progress meter!', 
# #                                i+1, 
# #                                NUM_FRAMES,
# #                                'Processing File '+str(k+1)+' of '+str(len(PATH))+': ',
# #                                os.path.split(PATH[k])[-1],
# #                                values[1:],
# #                                orientation='h')
# #     results.append(item)
    
# pool.close()
# pool.join()

# POSITIONS = pd.DataFrame(columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME', 'TIME'])
# for i in range(NUM_FRAMES):

#     X = results[i][0][0]
#     Y = results[i][1][0]
#     Z = results[i][2][0]
#     I_FS = results[i][3][0]
#     I_GS = results[i][4][0]
#     FRAME = i*np.ones_like(results[i][0][0])
#     TIME = FRAME / FRAME_RATE[k]
    
#     DATA = np.concatenate((np.expand_dims(X, axis=1), 
#                             np.expand_dims(Y, axis=1), 
#                             np.expand_dims(Z, axis=1), 
#                             np.expand_dims(I_FS, axis=1), 
#                             np.expand_dims(I_GS, axis=1), 
#                             np.expand_dims(FRAME, axis=1), 
#                             np.expand_dims(TIME, axis=1)), 
#                             axis=1)
#     # POSITIONS = POSITIONS.append(pd.DataFrame(DATA, 
#                                                 # columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME', 'TIME']))
    
#     POSITIONS = pd.concat([POSITIONS, pd.DataFrame(DATA, 
#                                                 columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME', 'TIME'])],
#                             ignore_index=True)

# POSITIONS = POSITIONS.astype('float')
# POSITIONS['TIME'] = POSITIONS['TIME'].round(2)

# st.dataframe(POSITIONS)

