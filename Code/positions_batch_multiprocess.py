#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:17:55 2020

@author: erick

Modified on Fri October 18 11:59:55 2024

@author: Maciek
"""

# Holographic Coordinate Detection

# Import libraries
import os
from multiprocessing import Pool, cpu_count
from time import time
from tqdm import tqdm

import PySimpleGUI as sg
import numpy as np
import pandas as pd
import functions as f

# Constant for opening File after export
OPEN_FILE = True

# GUI for selecting files and parameters
def main():
    layout = [
        [sg.Text('Select AVi File recording', size=(35, 1)),
         sg.FileBrowse(initial_folder='', key='-FILE-')],
        [sg.Text('Select Background Image', size=(35, 1)),
         sg.FileBrowse(initial_folder='', key='-BGIMAGE-')],
        [sg.Text('Propagator Scheme', size=(35, 1)),
         sg.Radio('Rayleigh-Sommerfeld', 'SCHEME', key='-RS-', default=True),
         sg.Radio('Modified', 'SCHEME', key='-MOD-')],
        [sg.Text('Refraction index of media (water = 1.3226)', size=(35, 1)),
         sg.InputText(default_text=1.3326, key='-N-')],
        [sg.Text('Wavelength in um (~0.642)', size=(35, 1)),
         sg.InputText(default_text=0.642, key='-WAVELENGTH-')],
        [sg.Text('Magnification (10, 20, etc)', size=(35, 1)),
         sg.InputText(default_text=40, key='-MPP-')],
        [sg.Text('Step size (10)', size=(35, 1)),
         sg.InputText(default_text=3, key='-SZ-')],
        [sg.Text('Number of steps (150)', size=(35, 1)),
         sg.InputText(default_text=30, key='-NUMSTEPS-')],
        [sg.Text('Gradient Stack Threshold (~0.1) just for RS', size=(35, 1)),
         sg.InputText(default_text=0.002, key='-THRESHOLD-')],
        [sg.Text('Peak Min Distance (20, 40, 60)', size=(35, 1)),
         sg.InputText(default_text=20, key='-PMD-')],
        [sg.Text('Frame Rate', size=(35, 1)),
         sg.InputText(default_text=50, key='-FRAMERATE-')],
        [sg.Checkbox('Invert Video', default=False, key='-INVERT-')],
        [sg.Checkbox('Export as CSV', default=True, key='-EXPORT-')],
        [sg.Text('Number of frames for calculations', size=(35, 1)),
         sg.InputText(default_text='50', key='-NUMFRAMES-')],
        [sg.Button('Add File'), sg.Button('Start'), sg.Cancel()]
    ]

    window = sg.Window('Holography video inputs', layout)

    # Initialize data storage
    paths, bg_paths, params = [], [], []

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, 'Cancel'):
            break
        elif event == 'Start':
            break
        elif event == 'Add File':
            print(f'File {os.path.split(values["-FILE-"])[-1]} added to queue')
            #paths.append(values['-FILE-'])
            #bg_paths.append(values['-BGIMAGE-'])
            paths.append(
                'C:/Users/mz1794/Downloads/Python and Viking DHM port-20241010T094616Z-001/Python and Viking DHM port/40x_100Hz_1081_CHO_1_T5_detrend_frame0-50.avi')
            bg_paths.append(
                'C:/Users/mz1794/Downloads/Python and Viking DHM port-20241010T094616Z-001/Python and Viking DHM port/1024bg.png')

            scheme = '-RS-' if values['-RS-'] else '-MOD-'
            params.append({
                'N': float(values['-N-']),
                'Wavelength': float(values['-WAVELENGTH-']),
                'MPP': int(values['-MPP-']),
                'SZ': float(values['-SZ-']),
                'NUMSTEPS': int(values['-NUMSTEPS-']),
                'THRESHOLD': float(values['-THRESHOLD-']),
                'PMD': int(values['-PMD-']),
                'FRAMERATE': int(values['-FRAMERATE-']),
                'INVERT': values['-INVERT-'],
                'EXPORT': values['-EXPORT-'],
                'NUMFRAMES': int(values['-NUMFRAMES-']) if values['-NUMFRAMES-'] else None,
                'SCHEME': scheme
            })

    window.close()
    print('-' * 150)

    # Position detection
    data = []
    times = []

    for k, path in enumerate(paths):
        vid = f.videoImport(path, 0)
        bg_image = f.bgPathToArray(bg_paths[k])
        ni, nj, nk = vid.shape

        if params[k]['INVERT']:
            for i in range(nk):
                vid[:, :, i] = vid[:, :, i].max() - vid[:, :, i]

        frames_median = 20
        i_median = f.medianImage(vid, frames_median)

        num_frames = params[k]['NUMFRAMES'] if params[k]['NUMFRAMES'] is not None else nk

        it = np.empty((num_frames), dtype=object)
        med = np.empty((num_frames), dtype=object)

        for i in range(num_frames):
            it[i] = vid[:, :, i]
            med[i] = i_median

        # Parameters setup
        n, lam, mpp, sz, numsteps = (params[k]['N'], params[k]['Wavelength'],
                                     params[k]['MPP'], params[k]['SZ'],
                                     params[k]['NUMSTEPS'])
        threshold, pmd = (params[k]['THRESHOLD'], params[k]['PMD'])

        pool = Pool(cpu_count())  # Number of cores to use
        results = []
        t0 = time()

        print(f'Processing File {k + 1} of {len(paths)}: {os.path.split(path)[-1]}')
        print('Parameters:', params[k])

        scheme_func = f.positions_batch if params[k]['SCHEME'] == '-RS-' else f.positions_batch_modified

        for _ in tqdm(pool.imap_unordered(scheme_func,
                                          zip(it, med, [n] * num_frames,
                                              [lam] * num_frames, [mpp] * num_frames,
                                              [sz] * num_frames, [numsteps] * num_frames,
                                              [threshold] * num_frames, [pmd] * num_frames,
                                              bg_image)), total=num_frames):
            results.append(_)

        times.append(time() - t0)
        pool.close()
        pool.join()

        positions = pd.DataFrame(columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME', 'TIME'])

        for i in range(num_frames):

            x = results[i][0][0]
            y = results[i][1][0]
            z = results[i][2][0]
            i_fs = results[i][3][0]
            i_gs = results[i][4][0]

            frame = np.full_like(x, i)
            time_values = frame / params[k]['FRAMERATE']

            data_row = np.column_stack((x, y, z, i_fs, i_gs, frame, time_values))
            positions = pd.concat([positions, pd.DataFrame(data_row,
                                                           columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME', 'TIME'])],
                                  ignore_index=True)

        positions = positions.astype('float')
        positions['TIME'] = positions['TIME'].round(3)

        # Export CSV Files
        if params[k]['EXPORT']:
            path = os.path.split(path)[:-1][0]
            pp = os.path.split(path)[-1][:-4]
            scheme_suffix = 'RS' if params[k]['SCHEME'] == '-RS-' else 'MOD'
            expath = f"{path}/{pp}_TH{params[k]['THRESHOLD']}_PMD{params[k]['PMD']}_SZ{params[k]['SZ']}_NUMSTEPS{params[k]['NUMSTEPS']}_{scheme_suffix}.csv"
            positions.to_csv(expath)

            print('Exported to:\n', expath)
            print('-' * 150)

            if OPEN_FILE:
                os.startfile(expath)

        data.append(positions)

    print('Done!')


if __name__ == '__main__':
    main()
