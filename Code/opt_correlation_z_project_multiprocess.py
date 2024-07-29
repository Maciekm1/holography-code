# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 05:23:04 2022

@author: eers500
"""

import numpy as np
import functions as f
import easygui as gui
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter, sobel
from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import tqdm
from skimage import restoration
import easygui as gui
from natsort import natsorted
import os
import cv2
import pickle
import cupy as cp
import pandas as pd

def analysis(array):
    window = 3                                          # Set by window in peak_gauss_fit_analysis() function
    w = 1                                               # Windos for quadratic fit in Z
    pol = lambda a, x: a[0]*x**2 + a[1]*x + a[2]
    pos = []

    temp = np.copy(array[0])
    nii, njj, nkk = temp.shape

    # methods = ['GPU', 'Optical']
    # method = methods[0]
    method = array[1]

    apply_filters = False  # Just for Optical

    
    if method == 'Optical':
        # temp = np.empty((nii, njj, mk))
        # ids = np.arange(k*mk, k*mk+mk) 
        
        # for i in range(nkk):
            # temp[:, :, i] = CC[id]
        
        if apply_filters:
            for i in range(nkk):
                temp[:, :, i] = gaussian_filter(np.abs(sobel(temp[:, :, i])), 1)
         
        zp = np.max(temp, axis=2)
        zp_gauss = gaussian_filter(zp.astype('float32'), sigma=1)
        # zp_gauss = zp
        
        
        # r = peak_local_max(zp_gauss.astype('float32'), threshold_rel=0.5, min_distance=50, num_peaks=1)
        r = peak_local_max(zp_gauss.astype('float32'), threshold_rel=0.8, min_distance=20)
    
    elif method == 'GPU':
        # temp = CC[:, :, k*mk:k*mk+mk]
        zp = np.max(temp, axis=2)
        zp_gauss = gaussian_filter(zp.astype('float32'), sigma=3)
        # zp_gauss = zp
        
        # r = peak_local_max(zp_gauss.astype('float32'), threshold_rel=0.2, min_distance=2, num_peaks=1)
        r = peak_local_max(zp_gauss.astype('float32'), threshold_rel=0.3, min_distance=20)
    
    for r0 in r: 
        ri, rj = r0[0], r0[1]
        zpp = temp[ri-window:ri+window, rj-window:rj+window, :]
        
        # zpp_sum = np.sum(zpp, axis=(0, 1))
        zpp_sum = np.max(zpp, axis=(0,1))
        # plt.plot(zpp_sum, '.-')
        
        idmax = np.where(zpp_sum == zpp_sum.max())[0][0]
        # filter_sel = np.where(zpp_sum == zpp_sum.max())[0][0]
        
        if idmax > w and idmax < nkk-w:
            ids = np.arange(idmax-w, idmax+w+1)
            ids_vals = zpp_sum[ids]
            coefs = np.polyfit(ids, np.float32(ids_vals), 2)
            
            interp_ids = np.linspace(ids[0], ids[-1], 100)
            interp_val = pol(coefs, interp_ids)
    
            filter_sel = interp_ids[interp_val == interp_val.max()][0] 
            # print(filter_sel)
        
        else:
            filter_sel = np.where(zpp_sum == zpp_sum.max())[0][0]
            # print(filter_sel)
        
        pos.append([ri, rj, filter_sel])

    locs = np.array(pos)

    return locs
#%%
if __name__ == '__main__':
             
    #% Video settings
    magnification = 40          # Archea: 20, E. coli: 40, 40, 40, MAY 20 (new 20), Colloids 20
    frame_rate = 60              # Archea: 30/5, E. coli: 60, 60, 60, MAY 100, Colloids 50
    fs = 0.711*(magnification/10)                  # px/um
    ps = (1 / fs)                    # Pixel size in image /um
    SZ = 5                     # step size of LUT [Archea: 10um,E. coli: 20, 40, 20, MAY 20 (new 10)], Colloids: 10
    number_of_images = 430      # Archea = 400 , Ecoli = 430, 430, 700  # MAY 275(550)
    number_of_filters = 25      # Archea =  25 ,   Ecoli =  19,  19,  20  # MAY 30 (new 40)  
    true_z_of_target_im_1 = 121.1 # 96.1
    first_time = False
    methods = ['GPU', 'Optical']
    method = methods[0]
    
    
    if first_time and method == 'Optical':
        #% Read correlation images file names OPTICAL ONLY
        path = gui.diropenbox()
        path = path + '/'
        
        file_list = os.listdir(path)
        
        need_sort = True
        if need_sort:
            file_list = natsorted(file_list)
        
        image_number = []
        filter_number = []
        for k in range(len(file_list)):
            filter_number.append(int(file_list[k][:2]))
            image_number.append(int(file_list[k][4:9]))
        
        image_number = np.array(image_number)
        filter_number = np.array(filter_number)
        
        #% Order arrays according to image-filter combination (iterate filter first) OPTICAL ONLY
        flist = []
        number = number_of_images*number_of_filters
        fnum = np.empty(number)
        inum = np.empty(number)
        filter_number = np.array(filter_number)
        image_number = np.array(image_number)
        
        for k in range(number):
            k_image_index = np.where(image_number == k)[0]
            
            for index in k_image_index:
                flist.append(file_list[index])
            
            fnum[k*number_of_filters:k*number_of_filters+number_of_filters] = filter_number[k_image_index]
            inum[k*number_of_filters:k*number_of_filters+number_of_filters] = image_number[k_image_index]
            
        file_list = flist
        
        #% Read images
        nk = number_of_images
        mk = number_of_filters
        
        print('Reading images...')
        
        CC = [cv2.imread(path+file, 0) for file in tqdm(file_list)]
        temps = [np.swapaxes(np.swapaxes(CC[k*mk:k*mk+mk], 0, 2), 0, 1) for k in range(nk)]
        
        #% Save as Pickle
        expath = gui.filesavebox()
        with open(expath, 'wb') as f:
            pickle.dump(temps, f)
    # elif not first_time and method == 'Optical':
    #     #% Read Pickle
    #     print('Reading Pickle File...')
    #     readpath = gui.fileopenbox()
    #     with open(readpath, 'rb') as f:
    #         temps = pickle.load(f)
            
    elif first_time and method == 'GPU':
        #%% Import Video correlate
        path_vid = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
        VID = f.videoImport(path_vid, 0)
        # VID = VID[:226, 300-226:,:]
        ni, nj, nk = np.shape(VID)

        invert = False

        if invert:
            for i in range(nk):
                VID[:, :, i] = VID[:, :, i].max() - VID[:, :, i]
                

        #% Import LUT form images
        path_lut = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
        LUT = f.videoImport(path_lut, 0)
        mi, mj, mk = np.shape(LUT)

        if invert:
            for i in range(mk):
                LUT[:, :, i] = LUT[:, :, i].max() - LUT[:, :, i]

        #% Prepare arrays
        VID_zn = np.empty_like(VID)
        for k in range(nk):
            A = VID[:,:,k]
            VID_zn[:, :, k] = (A-np.mean(A))/np.std(A)

        LUT_zn = np.empty_like(LUT)
        for k in range(mk):
            A = LUT[:,:,k]
            LUT_zn[:, :, k] = (A-np.mean(A))/np.std(A)

        #% CuPy correlation
        def corr_gpu(a, b):
            return a*cp.conj
        
        cFT = lambda x: cp.fft.fftshift(cp.fft.fft2(x))
        cIFT = lambda X: cp.fft.ifftshift(cp.fft.ifft2(X))
        cc = []
        CC = np.empty((ni, nj, nk*mk), dtype='float16')
        
        print('Computing correlation in GPU...')
        for i in tqdm(range(nk)):
        # for i in range(10):
            im = VID_zn[:, :, i]
            imft = cFT(cp.array(im))
            for j in range(mk):
                fm = cp.pad(cp.array(LUT_zn[:, :, j]), int((ni-mi)/2))
                fmft = cFT(fm)
                cc.append(cp.abs(cIFT(imft*cp.conj(fmft))).get().astype('float16'))
                # CC[:, :, i*mk+j] = np.abs(cIFT(corr_gpu(imft, fmft)))
                CC[:, :, i*mk+j] = cp.abs(cIFT(imft*cp.conj(fmft))).get().astype('float16')
        
        temps = [CC[:, :, k*mk:k*mk+mk] for k in range(nk)]
        
        #% Save as Pickle
        expath = gui.filesavebox()
        with open(expath, 'wb') as f:
            pickle.dump(temps, f)
     
    else:
        #% Read Pickle
        print('Reading Pickle File...')
        readpath = gui.fileopenbox()
        with open(readpath, 'rb') as f:
            temps = pickle.load(f)
        
    #%%
    pool = Pool(cpu_count())     # Number of cores to use
    results = []

    print('Processing File...')
    meth = len(temps)*[method]
    #tuple = (temps, meth)
    
    for _ in tqdm(pool.imap_unordered(analysis, zip(temps, meth)), total=len(temps)):
        results.append(_)
    
    #%
    locs = []
    for frame, loc in enumerate(results):
        if len(loc) != 0:
            a = np.hstack((loc, frame*np.ones((len(loc), 1))))
            a = pd.DataFrame(a, columns=['X', 'Y', 'Z', 'FRAME'])
            locs.append(a)
        
    locations = pd.concat(locs)
    locations['TIME'] = locations['FRAME']/frame_rate
    locations[['X', 'Y']] = locations[['X', 'Y']] * ps
    locations['Z'] = true_z_of_target_im_1 - locations['Z']
    
    plot = True
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
        POSITIONS = locations
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
        
    
    
    
    
    
    
    
