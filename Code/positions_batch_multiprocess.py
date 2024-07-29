#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:17:55 2020

@author: erick
"""

# # Holographic Coordinate Detection

#%% Import libraries
import os
import numpy as np
import PySimpleGUI as sg
import pandas as pd
import matplotlib.pyplot as plt
import functions as f
from time import time
from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import tqdm

#%% GUI for selecting files and parameters
if __name__ == '__main__':
    
    # %
    layout = [
        [sg.Text('Select AVi File recording', 
                 size=(35, 1)), 
                 # sg.In(default_text='Select a video'), 
                 sg.FileBrowse(initial_folder='/media/erick/NuevoVol/LINUX_LAP/PhD/Thesis/', key='-FILE-')],
        [sg.Text('Propagator Scheme', size=(35, 1)), sg.Radio('Rayleigh-Sommerfeld', 'SCHEME', key='-RS-'), sg.Radio('Modified', 'SCHEME', key='-MOD-')],
        [sg.Text('Refraction index of media (water = 1.3226)', size=(35, 1)), sg.InputText(default_text=1.3226, key='-N-')],
        [sg.Text('Wavelength in um (~0.642)', size=(35, 1)), sg.InputText(default_text=0.642, key='-WAVELENGTH-')],
        [sg.Text('Magnification (10, 20, etc)', size=(35, 1)), sg.InputText(default_text=10, key='-MPP-')],
        [sg.Text('Step size (10)', size=(35, 1)), sg.InputText(default_text=10, key='-SZ-')],
        [sg.Text('Number os steps (150)', size=(35, 1)), sg.InputText(default_text=150, key='-NUMSTEPS-')],
        [sg.Text('Gradient Stack Threshold (~0.1) just for RS', size=(35, 1)), sg.InputText(default_text=0.1, key='-THRESHOLD-')],
        [sg.Text('Peak Min Distance (20, 40, 60)', size=(35, 1)), sg.InputText(default_text=20, key='-PMD-')],
        [sg.Text('Frame Rate', size=(35, 1)), sg.InputText(default_text=50, key='-FRAMERATE-')],
        [sg.Checkbox('Invert Video', default=False, key='-INVERT-')],
        [sg.Checkbox('Export as CSV', default=True, key='-EXPORT-')],
        [sg.Text('Number of frames for calculations', size=(35, 1)), sg.InputText(default_text='', key='-NUMFRAMES-')],
        [sg.Button('Add File'), sg.Button('Start'), sg.Cancel()]
    ]
    
    window = sg.Window('Holography video inputs', layout)
    
    PATH = []
    N = []
    LAMBDA = []
    MPP = []
    SZ = []
    NUMSTEPS = []
    THRESHOLD = []
    PMD = []
    FRAME_RATE = []
    INVERT_VIDEO = []
    export = []
    num_frames = []
    scheme = []
    params = []
    
    while True:
        event, values = window.Read()
        # values = list(values.values())
        if event == 'Cancel': #or event == sg.WIN_CLOSED:
            break
        elif event == 'Start':
            break
        elif event == 'Add File':   
            # print('File '+os.path.split(values[0])[-1]+' added to queue')
            # PATH.append(values[0])
            # N.append(float(values[1]))
            # LAMBDA.append(float(values[2]))
            # MPP.append(int(values[3]))
            # SZ.append(float(values[4]))
            # NUMSTEPS.append(int(values[5]))
            # THRESHOLD.append(float(values[6]))
            # PMD.append(int(values[7]))
            # FRAME_RATE.append(int(values[8]))
            # INVERT_VIDEO.append(values[9])          
            # export.append(values[10])

            # if values[11] == '':
            #     num_frames.append([])
            # else:
            #     num_frames.append(int(values[11]))
            # params.append(pd.DataFrame([values[1:]], columns=['N', 'Wavelength', 'MPP', 'SZ', 'Step #', 'Threshold', 'MPD', 'FR','Invert', 'Export', 'Num Frames']))
            
            ##########################################################
            print('File '+os.path.split(values['-FILE-'])[-1]+' added to queue')
            PATH.append(values['-FILE-'])
            
            if values['-RS-'] == True:
                scheme.append('-RS-')
            else:
                scheme.append('-MOD-')
            
            
            N.append(float(values['-N-']))
            LAMBDA.append(float(values['-WAVELENGTH-']))
            MPP.append(int(values['-MPP-']))
            SZ.append(float(values['-SZ-']))
            NUMSTEPS.append(int(values['-NUMSTEPS-']))
            THRESHOLD.append(float(values['-THRESHOLD-']))
            PMD.append(int(values['-PMD-']))
            FRAME_RATE.append(int(values['-FRAMERATE-']))
            INVERT_VIDEO.append(values['-INVERT-'])          
            export.append(values['-EXPORT-'])
            
            if values['-NUMFRAMES-'] == '':
                num_frames.append([])
            else:
                num_frames.append(int(values['-NUMFRAMES-']))
            params.append(pd.DataFrame([list(values.values())[1:]], 
                                       # columns=['N', 'Wavelength', 'MPP', 'SZ', 'Step #', 'Threshold', 'MPD', 'FR','Invert', 'Export', 'Num Frames']
                                       columns=list(values.keys())[1:]
                                       )
                          )
            
            ##########################################################
            
    window.Close()
    print('---------------------------------------------------------------------------------------------------------------------------------------------------')
    
#%% Position detection
    
    data = []
    times = []
    for k in range(len(PATH)): 
        VID = f.videoImport(PATH[k], 0)
        ni, nj, nk = np.shape(VID)
        
        
        if INVERT_VIDEO[k]:
            for i in range(nk):
                VID[:, :, i] = VID[:, :, i].max() - VID[:, :, i]
    
        FRAMES_MEDIAN = 20
        I_MEDIAN = f.medianImage(VID, FRAMES_MEDIAN)
        # I_MEDIAN = np.ones((VID.shape[0], VID.shape[1]))
        
        if num_frames[k] == []:
            NUM_FRAMES = nk
        else:
            NUM_FRAMES = num_frames[k]
            
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
            IT[i] = VID[:, :, i]
            MED[i] = I_MEDIAN
            n[i] = N[k]
            lam[i] = LAMBDA[k]
            mpp[i] = MPP[k]
            sz[i] = SZ[k]
            numsteps[i] = NUMSTEPS[k]
            threshold[i] = THRESHOLD[k]
            pmd[i] = PMD[k]
        
        pool = Pool(cpu_count())     # Number of cores to use
        results = []
        T0 = time()
    
        print('Processing File '+str(k+1)+' of '+str(len(PATH))+': '+ os.path.split(PATH[k])[-1])
        print('Parameters: ')
        print(params[k])
        
        if scheme[k] == '-RS-':
            for _ in tqdm(pool.imap_unordered(f.positions_batch, zip(IT, MED, n, lam, mpp, sz, numsteps, threshold, pmd)), 
                          total=NUM_FRAMES):
                results.append(_)
        
        elif scheme[k] == '-MOD-':
            for _ in tqdm(pool.imap_unordered(f.positions_batch_modified, zip(IT, MED, n, lam, mpp, sz, numsteps, threshold, pmd)), 
                          total=NUM_FRAMES):
                results.append(_)
                
        times.append(time() - T0)   
        
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
        POSITIONS['TIME'] = POSITIONS['TIME'].round(3)

#%% Export CSV Files

        if export[k]:
            path = os.path.split(PATH[k])[:-1][0]
            pp = os.path.split(PATH[k])[-1][:-4]
            
            if scheme[k] == '-RS-':
                expath = path+'/'+pp+'_TH'+str(THRESHOLD[k])+'_PMD'+str(PMD[k])+'_SZ'+str(SZ[k])+'_NUMSTEPS'+str(NUMSTEPS[k])+'_RS'+'.csv'
                POSITIONS.to_csv(expath)
            
            elif scheme[k] == '-MOD-':
                expath = path+'/'+pp+'_TH'+str(THRESHOLD[k])+'_PMD'+str(PMD[k])+'_SZ'+str(SZ[k])+'_NUMSTEPS'+str(NUMSTEPS[k])+'_MOD'+'.csv'
                POSITIONS.to_csv(expath)
            print('Exported to: \n', expath)
            print('---------------------------------------------------------------------------------------------------------------------------------------------------')
               
        
    #     T = time.time - T0
    #     times.append(T)
        data.append(POSITIONS)
        
    print('Done!')
    
    plot = False
    if plot:
        #%% Scatter Plot
        import easygui as gui
        import pandas as pd
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot
        # # %matplotlib qt
        # # %matplotlib inline
        
        
        # path = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/Thesis/', multiple=False)
        # POSITIONS = pd.read_csv(path)
        POSITIONS = data[0]
        # # POSITIONS = DF
        
        X = POSITIONS.X
        Y = POSITIONS.Y
        Z = POSITIONS.Z
        T = POSITIONS.FRAME
        
        # # maxx = Z == Z.max()
        
        fig = pyplot.figure(1, figsize=(5, 5))
        ax = pyplot.axes(projection='3d')
        
        ax.scatter(X, Y, Z, s=1, marker='.', c=T)
        # #ax.plot(X, Y, Z)
        # # ax.scatter(X[maxx], Y[maxx], Z[maxx], c='red')
        ax.tick_params(axis='both', labelsize=10)
        ax.set_title('Cells Positions in 3D', fontsize='20')
        ax.set_xlabel('y (um)', fontsize='18')
        ax.set_ylabel('x (um)', fontsize='18')
        ax.set_zlabel('z (um)', fontsize='18')
        
        pyplot.show()

