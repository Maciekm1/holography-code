#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:11:21 2022

@author: erick
"""

class holography:
    
    #%% rgb2gray
    def rgb2gray(img):
        ## Convert rgb image to grayscale using Y' = 0.299R'+0.587G' + 0.114B'
        # Input:     img - RBG image
        # Output: img_gs - Grayscale image
        import numpy as np
        [ni, nj, nk] = img.shape
        img_gs = np.empty([ni, nj])
        for ii in range(0, ni):
            for jj in range(0, nj):
                img_gs[ii, jj] = 0.299 * img[ii, jj, 0] + 0.587 * img[ii, jj, 1] + 0.114 * img[ii, jj, 2]
    
        return img_gs
    
    
    #%% square_image
    def square_image(img):
        ## Make image square by adding rows or columns of the mean value of the image np.mean(img)
        # Input: img - grayscale image
        # Output: imgs - square image
        #         axis - axis where data is added
        #            d - number of rows/columns added
        import numpy as np
    
        [ni, nj] = img.shape
        dn = ni - nj
        d = abs(dn)
        if dn < 0:
            M = np.flip(img[ni - abs(dn):ni, :], 0)
            imgs = np.concatenate((img, M), axis=0)
            axis = 'i'
        elif dn > 0:
            M = np.flip(img[:, nj - abs(dn):nj], 1)
            imgs = np.concatenate((img, M), axis=1)
            axis = 'j'
        elif dn == 0:
            imgs = img
            axis = 'square'
        return imgs, axis, d
    
    
    #%% bandpassFilter
    def bandpassFilter(img, xs, xl):
        ## Bandpass filter
        # Input: img - Grayscale image array (2D)
        #        xl  - Large cutoff size (Pixels)
        #        xs  - Small cutoff size (Pixels)
        # Output: img_filt - filtered image
        import numpy as np
    
        # FFT the grayscale image
        imgfft = np.fft.fft2(img)
        img_fft = np.fft.fftshift(imgfft)
        img_amp = abs(img_fft)
        del imgfft
    
        # Pre filter image information
        [ni, nj] = img_amp.shape
        MIS = ni
    
        # Create bandpass filter when BigAxis ==
        # LCO = np.empty([ni, nj])
        # SCO = np.empty([ni, nj])
    
        # for ii in range(ni):
        #     for jj in range(nj):
        #         LCO[ii, jj] = np.exp(-((ii - MIS / 2) ** 2 + (jj - MIS / 2) ** 2) * (2 * xl / MIS) ** 2)
        #         SCO[ii, jj] = np.exp(-((ii - MIS / 2) ** 2 + (jj - MIS / 2) ** 2) * (2 * xs / MIS) ** 2)
        # BP = SCO - LCO
    
        jj, ii = np.meshgrid(np.arange(nj), np.arange(ni))
        
        LCO = np.exp(-((ii-MIS/2)**2 + (jj-MIS/2)**2) * (2*xl/MIS)**2)
        SCO = np.exp(-((ii-MIS/2)**2 + (jj-MIS/2)**2) * (2*xs/MIS)**2)
        BP =  SCO - LCO
        BPP = np.fft.ifftshift(BP)
        
        # Filter image
        filtered = BP * img_fft
        img_filt = np.fft.ifftshift(filtered)
        img_filt = np.fft.ifft2(img_filt)
        # img_filt = np.rot90(np.real(img_filt),2)
    
        return img_filt, BPP
    
    
    #%% videoImport
    def videoImport(video, N):
        ## Import video as stack of images in a 3D array
        #   Input:  video   - path to video file
        #               N   - frame number to import
        #   Output: imStack - 3D array of stacked images in 8-bit
        import cv2
        import numpy as np
    
        CAP = cv2.VideoCapture(video)
        NUM_FRAMES = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
        WIDTH = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
        HEIGHT = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        # IMG = np.empty((NUM_FRAMES, HEIGHT, WIDTH, 3), np.dtype('uint8'))
        # IM_STACK = np.empty((NUM_FRAMES, HEIGHT, WIDTH))
    
        I = 0
        SUCCESS = True
    
        if N == 0:
            # IMG = np.empty((NUM_FRAMES, HEIGHT, WIDTH, 3), dtype='float16')
            IMG = np.empty((HEIGHT, WIDTH, 3))
            IM_STACK = np.empty((NUM_FRAMES, HEIGHT, WIDTH), dtype='float32')
    
            while (I < NUM_FRAMES and SUCCESS):
                SUCCESS, IMG = CAP.read()
                # IM_STACK[I] = IMG[I, :, :, 1]
                IM_STACK[I] = IMG[:, :, 0]
                I += 1
                # print(('VI', I))
    
        elif N > 0:
            IMG = np.empty((NUM_FRAMES, HEIGHT, WIDTH, 3), dtype='float32')
            IM_STACK = np.empty((NUM_FRAMES, HEIGHT, WIDTH))
            STACK = IM_STACK
    
            while (I < NUM_FRAMES and SUCCESS):
                SUCCESS, IMG[I] = CAP.read()
                STACK[I] = IMG[I, :, :, 1]
                if I == N:
                    IM_STACK = IMG[I, :, :, 1]
                    FRAMENUM = I
                    print(('VI', I))
                I += 1
        CAP.release()
    
        if N == 0:
            IM_STACK = np.swapaxes(np.swapaxes(IM_STACK, 0, 2), 0, 1)
    
        return IM_STACK
    
    
    #%% exportAVI
    def exportAVI(filename, IM, NI, NJ, fps):
        ## Export 3D array to .AVI movie file
        #   Input:  IM - numpy 3D array
        #           NI - number of rows of array
        #           NJ - number of columns of array
        #          fps - frames per second of output file
        #   Output: .AVI file in working folder
        import os
        import numpy as np
        from cv2 import VideoWriter, VideoWriter_fourcc
        
        dir = os.getcwd()
        filenames = os.path.join(dir, filename)
        FOURCC = VideoWriter_fourcc(*'MJPG')
        VIDEO = VideoWriter(filenames, FOURCC, float(fps), (NJ, NI), 0)
    
        for i in range(IM.shape[2]):
            frame = IM[:, :, i]
            frame = np.uint8(255*frame/frame.max())
            #    frame = np.random.randint(0, 255, (NI, NJ,3)).astype('uint8')
            VIDEO.write(frame)
    
        VIDEO.release()
    
        print(filename, 'exported successfully')
        return
    
    
    #%% rayleighSommerfeldPropagator
    def rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, bandpass, med_filter):
        ## Rayleigh-Sommerfeld Back Propagator
        #   Inputs:          I - hologram (grayscale)
        #             I_MEDIAN - median image
        #                    Z - numpy array defining defocusing distances
        #   Output:        IMM - 3D array representing stack of images at different Z
        import numpy as np
        from functions import bandpassFilter
        from scipy.ndimage import median_filter
    
        # Divide by Median image
        I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)
        IN = I / I_MEDIAN
        if med_filter:
            IN = median_filter(IN, size=1)
        
        #    IN = I - I_MEDIAN,
        #     IN[IN < 0] = 0
    
        if bandpass:
            _, BP = bandpassFilter(IN, 2, 30)
            E = np.fft.fftshift(BP) * np.fft.fftshift(np.fft.fft2(IN - 1))
        else:
            E = np.fft.fftshift(np.fft.fft2(IN - 1))
    
    
        # Patameters     #Set as input parameters
        # N = 1.3226               # Index of refraction
        LAMBDA = LAMBDA       # HeNe
        FS = FS               # Sampling Frequency px/um
        NI = np.shape(IN)[0]  # Number of rows
        NJ = np.shape(IN)[1]  # Nymber of columns
        # SZ = 10
        Z = SZ*np.arange(0, NUMSTEPS)
        # Z = (FS * (51 / 31)) * np.arange(0, NUMSTEPS)
        #    Z = SZ*np.arange(0, NUMSTEPS)
        K = 2 * np.pi * N / LAMBDA  # Wavenumber
    
        # Rayleigh-Sommerfeld Arrays
        #P = np.empty_like(I_MEDIAN, dtype='complex64')
    
        #for i in range(NI):
        #    for j in range(NJ):
        #        P[i, j] = ((LAMBDA * FS) / (max([NI, NJ]) * N)) ** 2 * ((i - NI / 2) ** 2 + (j - NJ / 2) ** 2)
    
        jj, ii = np.meshgrid(np.arange(NJ), np.arange(NI))
        const = ((LAMBDA*FS)/(max([NI, NJ])*N))**2
        P = const*((ii-NI/2)**2 + (jj-NJ/2)**2)
    
        P = np.conj(P)
        Q = np.sqrt(1 - P) - 1
    
        if all(Z > 0):
            Q = np.conj(Q)
    
        # R = np.empty([NI, NJ, Z.shape[0]], dtype=complex)
        IZ = np.empty([NI, NJ, Z.shape[0]], dtype='float32')
    
        for k in range(Z.shape[0]):
            R = np.exp((-1j*K*Z[k]*Q), dtype='complex64')
            IZ[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E*R)))
        #        print(('RS', k))
        return IZ
    
    
    #%% medianImage
    def medianImage(VID, numFrames):
        ## Median Image
        #   Input:   VID - 3D numpy array of video file
        #            numFrames - Number of frames to calculat median image
        #   Output: MEAN - 2D pixel mean array
        import numpy as np
    
        def spaced_elements(array, numElems):
            out = array[np.round(np.linspace(0, len(array) - 1, numElems)).astype(int)]
            return out
    
        N = np.shape(VID)[2]
        id = spaced_elements(np.arange(N), numFrames)
    
        # print('MI')
        MEAN = np.median(VID[:, :, id], axis=2)
    
        return MEAN
    
    
    #%% zGradientStack
    def zGradientStack(IM):
        # Z-Gradient Stack
        #   Inputs:   I - hologram (grayscale)
        #            IM - median image
        #             Z - numpy array defining defocusing distances
        #   Output: CONV - 3D array representing stack of images at different Z
        import numpy as np
        from scipy import ndimage
        from functions import rayleighSommerfeldPropagator, exportAVI
    
        #    I = mpimg.imread('131118-1.png')
        #    I_MEDIAN = mpimg.imread('AVG_131118-2.png')
        #    Z = 0.02*np.arange(1, 151)
        #     IM = rayleighSommerfeldPropagator(I, I_MEDIAN, Z)
    
        #% Sobel-type kernel
        SZ0 = np.array(([-1, -2, -1], [-2, -4, -2], [-1, -2, -1]), dtype='float')
        SZ1 = np.zeros_like(SZ0)
        SZ2 = -SZ0
        SZ = np.stack((SZ0, SZ1, SZ2), axis=2)
        del SZ0, SZ1, SZ2
    
        # Convolution IM*SZ
        # IM = IM ** 2
        IMM = np.dstack((IM[:, :, 0][:, :, np.newaxis], IM, IM[:, :, -1][:, :, np.newaxis]))
        GS = ndimage.convolve(IMM, SZ, mode='mirror')
        GS = np.delete(GS, [0, np.shape(GS)[2] - 1], axis=2)
        del IMM
    
        #    exportAVI('gradientStack.avi',CONV, CONV.shape[0], CONV.shape[1], 24)
        #    exportAVI('frameStack.avi', IM, IM.shape[0], IM.shape[1], 24)
        return GS
    
    
    #%% dataCursor1D
    def dataCursor1D():
        # Data Cursor in plots
        import mpldatacursor
        mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'),
                                 formatter='x = {i}\ny = {y:.06g}'.format)
        return 0
    
    
    #%% dataCursor2D
    def dataCursor2D():
        # Data Cursor in 2D plots
        import mpldatacursor
        mpldatacursor.datacursor(display='multiple', hover=True, bbox=dict(alpha=1, fc='w'),
                                 formatter='x, y = {i}, {j}\nz = {z:.06g}'.format)
        return 0
    
    
    #%% dataCursor3D
    def dataCursor3D():
        # Data Cursor in 3D plots
        import mpldatacursor
        mpldatacursor.datacursor(hover=False, bbox=dict(alpha=1, fc='w'),
                                 formatter='x, y = {i}, {j}\nz = {z:.06g}'.format)
        return 0
    
    
    #%% histequ
    def histeq(im):
        ## Histogram equalization of a grayscale image
        import numpy as np
        from PIL import Image
    
        # get image histogram
        imhist, bins = np.histogram(im.flatten(), 256, normed=True)
        cdf = imhist.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize
    
        # use linear interpolation of cdf to find new pixel values
        im2 = np.interp(im.flatten(), bins[:-1], cdf)
    
        return im2.reshape(im.shape), cdf
    
    
    #%% guiImport
    def guiImport():
        # GUI for values
        import PySimpleGUI as sg
    
        layout = [
            [sg.Text('Select AVi File recording', size=(35, 1)), sg.In(), sg.FileBrowse()],
            [sg.Text('Select Median Image (optional)', size=(35, 1)), sg.In(), sg.FileBrowse()],
            [sg.Text('Frame number to use for calculations', size=(35, 1)), sg.InputText(default_text=1)],
            [sg.Text('Number of frames for median image', size=(35, 1)), sg.InputText(default_text=20)],
            [sg.Text('Refraction index of media (water = 1.3226)', size=(35, 1)), sg.InputText(default_text=1.3226)],
            [sg.Text('Wavelength in um (HeNe/0.642)', size=(35, 1)), sg.InputText(default_text=0.642)],
            [sg.Text('Sampling Frequency px/um (0.711) ', size=(35, 1)), sg.InputText(default_text=0.711)],
            [sg.Text('Step size (10)', size=(35, 1)), sg.InputText(default_text=10)],
            [sg.Text('Number os steps (150)', size=(35, 1)), sg.InputText(default_text=150)],
            [sg.Text('Gradient Stack Threshold (~0.1)', size=(35, 1)), sg.InputText(default_text=0.1)],
            [sg.Text('Magnification (10, 20, etc)', size=(35, 1)), sg.InputText(default_text=10)],
            [sg.Submit(), sg.Cancel()]
        ]
    
        window = sg.Window('Hologramphy inputs', layout)
        event, values = window.Read()
        window.Close()
    
        return values
    
    
    #%% Positions3D
    # Particles positions in 3D
    def positions3D(GS, peak_min_distance, num_particles, MPP):
        import numpy as np
        from skimage.feature import peak_local_max
       
        ZP = np.max(GS, axis=-1)
        if num_particles == 'None':
            PKS = peak_local_max(ZP, min_distance=peak_min_distance)  # 30
        elif num_particles > 0:
            PKS = peak_local_max(ZP, min_distance=peak_min_distance, num_peaks=num_particles)
        
        # import matplotlib.pyplot as plt
        # plt.imshow(ZP, cmap='gray')
        # plt.scatter(PKS[:,1], PKS[:,0], marker='o', facecolors='none', s=80, edgecolors='r')
        # plt.show()
        
        D1 = int(MPP/10)
        D2 = int(MPP/10)
        Z_SUM_XY = np.empty((GS.shape[2], len(PKS)))
        for ii in range(len(PKS)):
            idi = PKS[ii, 0]
            idj = PKS[ii, 1]
            A = GS[idi-D1:idi+D2:, idj-D1:idj+D2, :]                # How to treat borders?
            Z_SUM_XY[:, ii] = np.sum(A, axis=(0, 1))
        
        Z_SUM_XY_MAXS_FOLDED = np.empty((len(PKS), 1), dtype=object)
        for ii in range(len(PKS)):
            Z_SUM_XY_MAXS_FOLDED[ii, 0] = peak_local_max(Z_SUM_XY[:, ii], num_peaks=1)
            if Z_SUM_XY_MAXS_FOLDED[ii, 0].size == 0:
                Z_SUM_XY_MAXS_FOLDED[ii, 0] = np.array([[0]])
        
        
    
        Z_SUM_XY_MAXS = []
        for ii in range(len(Z_SUM_XY_MAXS_FOLDED)):
            if len(Z_SUM_XY_MAXS_FOLDED[ii, 0]) != 1:
                for jj in range(len(Z_SUM_XY_MAXS_FOLDED[ii, 0])):
                    Z_SUM_XY_MAXS.append([Z_SUM_XY_MAXS_FOLDED[ii, 0][jj].item()])
            else:
                Z_SUM_XY_MAXS.append([Z_SUM_XY_MAXS_FOLDED[ii, 0].item()])
        
        Z_SUM_XY_MAXS = np.array(Z_SUM_XY_MAXS)
    
        # Use Z_SUM_XY and Z_SUM_XY_MAXS
        w = 2 #2
        pol = lambda a, x: a[0]*x**2 + a[1]*x + a[2]
        z_max = []
        
        for j in range(len(Z_SUM_XY_MAXS)):
            
            i = Z_SUM_XY_MAXS[j][0]
            idi = np.arange(i-w,i+w+1)
            temp = np.pad(Z_SUM_XY, ((w, w), (0, 0)))
            val = temp[idi+w, j]
                    
            coefs = np.polyfit(idi, val, 2)
            
            interp_idi = np.linspace(idi[0], idi[-1], 10)
            interp_val = pol(coefs, interp_idi)
            
            idi_max = np.where(interp_val == interp_val.max())[0][0]
            z_max.append(interp_idi[idi_max])
        
        # XYZ_POSITIONS = np.hstack((XYZ_POSITIONS, Z_SUM_XY_MAXS[:, 0]))    # YXZ_POSITIONS = np.insert(PKS, 2, Z_SUM_XY_MAXS[:, 0], axis=-1)         # Actually [Y, X, Z]
        YXZ_POSITIONS = np.insert(np.float16(PKS), 2, z_max, axis=-1) 
    
        return YXZ_POSITIONS   # (x,y) in pixels, z in slice number
    
    
    #%% plot3D
    def plot3D(LOCS, title, fig, ax):
        # 3D Scatter Plot
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot
    
        # fig = pyplot.figure()
        # ax = Axes3D(fig)
    
        ax.scatter(LOCS[:, 0], LOCS[:, 1], LOCS[:, 2], s=25, marker='o')
        ax.tick_params(axis='both', labelsize=10)
        ax.set_title(title, fontsize='20')
        ax.set_xlabel('x (pixels)', fontsize='18')
        ax.set_ylabel('y (pixels)', fontsize='18')
        ax.set_zlabel('z (slices)', fontsize='18')
        pyplot.show()
    
        return
    
    #%% imshow_sequence
    def imshow_sequence(im, delay, run):
        import matplotlib.pyplot as plt
        import numpy as np
    
        fig, ax = plt.subplots()
    
        while run:
            for i in range(np.shape(im)[2]):
                ax.cla()
                ax.imshow(im[:, :, i], cmap='gray')
                ax.set_title("frame {}".format(i))
                # Note that using time.sleep does *not* work here!
                plt.pause(delay)
    
        return
    
    #%% fraunhoffer
    # Fraunhofer diffraction
    def fraunhofer(rho, wsize, zdist):
        """Fraunhofer diffraction"""
        import numpy as np
        from scipy.special import j1
        lam = 0.642     # lambda
        # z = 5
        return j1(np.pi*wsize*rho/(lam*zdist))/(wsize*rho/(lam*zdist))
    
    #%% Imshow_slider
    def imshow_slider(cube, axis, color_map):
        """
        Display a 3d ndarray with a slider to move along the third dimension.
    
        Extra keyword arguments are passed to imshow
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button, RadioButtons
    
        # check dim
        if not cube.ndim == 3:
            raise ValueError("cube should be an ndarray with ndim == 3")
    
        # generate figure
        fig = plt.figure()
        ax = plt.subplot(111)
        fig.subplots_adjust(left=0.25, bottom=0.25)
    
        # select first image
        # s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
        
        if axis == 0:
            im = cube[0, :, :]
        elif axis == 1:
            im = cube[:, 0, :]
        elif axis == 2:
            im = cube[:, :, 0]
    
        # display image
        l = ax.imshow(im, cmap=color_map)
    
        # define slider
        ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    
        slider = Slider(ax, 'Axis %i index' % axis, 0, cube.shape[axis] - 1,
                        valinit=0, valfmt='%i')
    
        def update(val):
            ind = int(slider.val)
            s = [slice(ind, ind + 1) if i == axis else slice(None)
                     for i in range(3)]
            im = cube[s].squeeze()
            l.set_data(im)
            fig.canvas.draw()
    
        slider.on_changed(update)
    
        plt.show()
        
    #%% modified_propagator
    def modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS):
        ## Rayleigh-Sommerfeld Back Propagator
        #   Inputs:          I - hologram (grayscale)
        #             I_MEDIAN - median image
        #                    Z - numpy array defining defocusing distances
        #   Output:        IMM - 3D array representing stack of images at different Z
        import numpy as np
        from functions import bandpassFilter, histeq
    
        # Divide by Median image
        I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)
        IN = I / I_MEDIAN
        #    IN = I - I_MEDIAN,
        #     IN[IN < 0] = 0
    
        # Bandpass Filter
        _, BP = bandpassFilter(IN, 2, 30)
        E = np.fft.fftshift(BP) * np.fft.fftshift(np.fft.fft2(IN - 1))
    
        # Patameters     #Set as input parameters
        # N = 1.3226               # Index of refraction
        LAMBDA = LAMBDA       # HeNe
        FS = FS               # Sampling Frequency px/um
        NI = np.shape(IN)[0]  # Number of rows
        NJ = np.shape(IN)[1]  # Nymber of columns
        # SZ = 10
        Z = SZ*np.arange(0, NUMSTEPS)
        # Z = (FS * (51 / 31)) * np.arange(0, NUMSTEPS)
        #    Z = SZ*np.arange(0, NUMSTEPS)
        K = 2 * np.pi * N / LAMBDA  # Wavenumber
    
        # Rayleigh-Sommerfeld Arrays
        # Q = np.empty_like(I_MEDIAN, dtype='complex64')
        # for i in range(NI):
        #     for j in range(NJ):
        #         Q[i, j] = ((LAMBDA * FS) / (max([NI, NJ]) * N)) ** 2 * ((i - NI / 2) ** 2 + (j - NJ / 2) ** 2)
    
        jj, ii = np.meshgrid(np.arange(NJ), np.arange(NI))
        const = ((LAMBDA*FS)/(max([NI, NJ])*N))**2
        Q = const*((ii-NI/2)**2 + (jj-NJ/2)**2)
    
        # P = np.conj(P)
        Q = np.sqrt(1 - Q) - 1
    
        if all(Z > 0):
            Q = np.conj(Q)
    
        R = np.empty([NI, NJ, Z.shape[0]], dtype='complex64')
        GS = np.empty([NI, NJ, Z.shape[0]], dtype='float32')
        # R1 = np.empty([NI, NJ, Z.shape[0]], dtype='complex64')
        # IZ = np.empty([NI, NJ, Z.shape[0]], dtype='float32')
    
        for k in range(Z.shape[0]):
            R = 1j*K*Q*np.exp((1j*K*Z[k]*Q), dtype='complex64')
            # R1 = np.exp((-1j*K*Z[k]*Q), dtype='complex64')
            GS[:, :, k] = np.abs(1 + np.fft.ifft2(np.fft.ifftshift(E*R)))
            # IZ[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E * R1)))
            
        # GS, _ = histeq(GS)
        # TH = 254.9
        # GS[GS < TH] = 0
        # GS = 255*((GS - TH) / np.max(GS - TH))
        # GS[GS < 250] = 0
        
        GS = GS - 1
        _, BINS = np.histogram(GS.flatten())
        GS[GS < BINS[9]] = 0
    
        return GS
    
    #%% Smooth trajectories piecewise cubic spline
    def smooth_curve(L, spline_degree, lim, sc):
        import numpy as np
        from scipy import interpolate
        # from mpl_toolkits.mplot3d import Axes3D
        from scipy import ndimage
        
        #%
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # L = LINKED[LINKED.PARTICLE == 6].values
            
        num_sample_pts = len(L)
        x_sample = L[:, 0]
        y_sample = L[:, 1]
        z_sample = L[:, 2]
        
        jump = np.sqrt(np.diff(x_sample)**2 + np.diff(y_sample)**2 + np.diff(z_sample)**2) 
        smooth_jump = ndimage.gaussian_filter1d(jump, 1, mode='wrap')  # window of size 5 is arbitrary
        limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
        xn, yn, zn = x_sample[:-1], y_sample[:-1], z_sample[:-1]
        # xn = xn[(jump > 0) & (smooth_jump < limit)]
        # yn = yn[(jump > 0) & (smooth_jump < limit)]
        # zn = zn[(jump > 0) & (smooth_jump < limit)]
        
        xn = xn[(jump > 0)]
        yn = yn[(jump > 0)]
        zn = zn[(jump > 0)]
        
        m = len(xn)
        smoothing_condition = (m-np.sqrt(m), m+np.sqrt(m))
        smoothing_condition = np.mean(smoothing_condition)
        # smoothing_condition = sc
        spline_degree = 3
        
        if len(xn) > 3:
            tck, u = interpolate.splprep([xn,yn,zn], s=smoothing_condition, k=spline_degree)
            x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
            u_fine = np.linspace(0, 1, num_sample_pts)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            X = [x_fine, y_fine, z_fine]
        else:
            X = -1
            
        # fig = plt.figure(figsize=(7, 4.5))
        # ax1 = fig.add_subplot(111, projection='3d')
        # ax1.set_facecolor('none')
        # ax1.plot(x_sample, y_sample, z_sample, 'o')
        # ax1.plot(x_fine, y_fine, z_fine, '-')
        # plt.show()
        
        #%        
        return X
    
    #%% smoothing with CSAPS
    def csaps_smoothing(L, smoothing_condition, filter_data, limit):
        import numpy as np
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        from scipy import ndimage
        from csaps import csaps, CubicSmoothingSpline
        
        N = len(L)
        # L = LINKED[LINKED.PARTICLE == 2]
        # t_sample = np.linspace(0, 1, len(L))
        t_sample = L['TIME'].values
        t_frames = L['FRAME'].values
        x_sample = L['X'].values
        y_sample = L['Y'].values
        z_sample = L['Z'].values
        data = [x_sample, y_sample, z_sample]
        # t_interp = np.linspace(0, 1, 1*len(L))
        # t_interp = t_sample
        t_interp = np.linspace(t_sample[0], t_sample[-1], 4*N)
        
        # smoothing_condition = 0.1
        
        if filter_data == False:
            # Smooth sample data
            smooth_data = csaps(t_sample, data, t_interp, smooth=smoothing_condition)
            x_smooth = smooth_data[0, :]
            y_smooth = smooth_data[1, :]
            z_smooth = smooth_data[2, :]
            tn = t_interp
            tframe = t_frames
            
            # # Smooth sample data with variable smoothing condition
            # xi, smooth_x = csaps(t_sample, x_sample, t_interp)
            # yi, smooth_y = csaps(t_sample, y_sample, t_interp)
            # zi, smooth_z = csaps(t_sample, z_sample, t_interp)
        elif filter_data == True:
            # Filter sample data
            jump = np.sqrt(np.diff(x_sample)**2 + np.diff(y_sample)**2 + np.diff(z_sample)**2) 
            smooth_jump = ndimage.gaussian_filter1d(jump, 2, mode='wrap')  # window of size 5 is arbitrary
            # limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
            # limit = smooth_jump.max()*0.2
            xn, yn, zn, tn, tframe = x_sample[:-1], y_sample[:-1], z_sample[:-1], t_sample[:-1], t_frames[:-1]
            xn = xn[(jump > 0) & (smooth_jump < limit)]
            yn = yn[(jump > 0) & (smooth_jump < limit)]
            zn = zn[(jump > 0) & (smooth_jump < limit)]
            # tn = np.linspace(0, 1, len(zn))
            tn = tn[(jump > 0) & (smooth_jump < limit)]
            tframe = tframe[(jump > 0) & (smooth_jump < limit)]
            
            # Smooth filtered data
            datani_smooth = csaps(tn, [xn, yn, zn], tn, smooth=smoothing_condition)
            x_smooth = datani_smooth[0, :]
            y_smooth = datani_smooth[1, :]
            z_smooth = datani_smooth[2, :]
            
            # xni_smooth = datani_smooth[0, :]
            # yni_smooth = datani_smooth[1, :]
            # zni_smooth = datani_smooth[2, :]
            
            # # Smooth filtered sample data with variable smoothing condition
            # xni, smooth_xni = csaps(tn, xn, tn)
            # yni, smooth_yni = csaps(tn, yn, tn)
            # zni, smooth_zni = csaps(tn, zn, tn)
        
        
        return [x_smooth, y_smooth, z_smooth, tn]
        
        
    #%% Surface plot
    
    def surf(array):
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt
        import numpy as np
    
        x = np.arange(array.shape[1])
        y = np.arange(array.shape[0])
        X, Y = np.meshgrid(x, y)
        
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, array, cmap='jet')
        plt.show()
    
    
        return
    
    #%% Speed Analysis
    # s : Data frame of [X, Y, Z, TIME]
    # returns speed, x, y, z
    def get_speed(s):
        import numpy as np
        time = s['TIME'].values
        x = s['X'].values
        y = s['Y'].values
        z = s['Z'].values
        
        DT = np.diff(time)
        vx = np.diff(x) / DT
        vy = np.diff(y) / DT
        vz = np.diff(z) / DT
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        
        return v, x[:-1], y[:-1], z[:-1], time[:-1]
    
    
    #%% Contiguous repeats
    
    def contiguous_repeats(array):
        import numpy as np
        
        x_b = 0*np.ones_like(array)
    
        i = 0
        while i < len(array):
            repeats = 1
            val = array[i]
            for j in range(1, len(array)-i):
                if array[i+j] == val:
                    repeats = repeats + 1
                else:
                    break
            x_b[i:i+j+1] = repeats
            i = i+repeats
            
        return x_b
    
    #%% Clean LINKED tracks
    def clean_tracks(LL):
        import numpy as np
        from functions import contiguous_repeats
        
        t = LL['TIME'].values
        fr = LL['FRAME'].values
        p = LL['PARTICLE'].values
        x = LL['X'].values
        y = LL['Y'].values
        z = LL['Z'].values
        r = np.sqrt(x**2+y**2+z**2)
    
        t_reps = contiguous_repeats(t)
        reps_vals = set(t_reps)
        reps_vals = [val for val in reps_vals if val > 1]
    
        if reps_vals == []:
            # print('No repeated values in time for PARTICLE = ', p[0])
            xx, yy, zz, tt, ff, pp = x, y, z, t, fr, p
    
        else:
            # print('There are repeated values in time for PARTICLE =', p[0])
            # for i, t_rep in enumerate(t_reps):
            
            if t_reps[0] > 1:
                for i in range(1, int(t_reps[0])):
                    t[i] = np.nan
                    # print(i)
    
                i = int(t_reps[0])
                t_rep = t_reps[i]
            else:
                i=0
                t_rep = t_reps[i]
    
            while i < len(t_reps)-1 and i+t_rep >= len(t_reps):
                    
                if t_rep > 1:
                    ids = np.arange(t_rep, dtype='int64')+i
                    ref_pos = r[i-1]
                    r_pos = r[ids]
                    dr = np.abs(ref_pos-r_pos)
                    idmin = np.argmin(dr)+i
                    idpop = ids[ids != idmin]
                    
                    for id in idpop:
                        t[idpop] = np.nan
    
                    # if i+t_rep >= len(t_reps):
                    #     break
                    else:
                        i = i+t_rep
                        t_rep = t_reps[i]
    
                else:
                    i= i+1
                    t_rep = t_reps[i]
                    
                        
            boolnan = np.isnan(t)
            tt = t[~boolnan]
            ff = fr[~boolnan]
            xx = x[~boolnan]
            yy = y[~boolnan]
            zz = z[~boolnan]
            pp = p[~boolnan]
        
        
        return [xx, yy, zz, tt, ff, pp]
    
    #%% MSD
    def MSD(x, y, z):
        import numpy as np
        from scipy.optimize import curve_fit
        
        ntaus = np.floor(len(x)/2)
        taus = np.arange(1, ntaus, dtype='uint64')
        MSD = []
        size = []
        for tau in taus:
            tau = int(tau)
            length = int(len(x)-tau)
            diff = np.empty(length)
            for i in range(length):
                diff[i] = (x[i]-x[i+tau])**2 + (y[i]-y[i+tau])**2 + (z[i]-z[i+tau])**2
            
            size.append(len(diff))
            MSD.append(np.mean(diff))
    
        def line(x, m, b):
            return m*x + b
    
        # def fun(t, v, D):
        #     return t**2*v**2 + 4*D*t 
    
        n = int(len(taus)/4)
        # n = len(taus)
        
        params, cov = curve_fit(line, taus[:n], MSD[:n], p0=(1, 0))
        # params, cov = curve_fit(fun, taus[:n], MSD[:n])
        
        if params[0] > 1.5:
            swim = True
        else:
            swim = False
            
        return MSD, swim
    
    #%% Positions batch
    def positions_batch(TUPLE):
        import numpy as np
        import functions as f
        I = TUPLE[0]
        I_MEDIAN = TUPLE[1]    
        N = TUPLE[2]
        LAMBDA = TUPLE[3]
        MPP = TUPLE[4]
        FS = (MPP/10)*0.711
        SZ = TUPLE[5]
        NUMSTEPS = TUPLE[6]
        THRESHOLD = TUPLE[7]
        PMD = TUPLE[8]
        
        LOCS = np.empty((1, 3), dtype=object)
        X, Y, Z, I_FS, I_GS = [], [] ,[], [], []
        IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, True, False).astype('float32')
        GS = f.zGradientStack(IM).astype('float32')  
        # GS = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS)  # Modified propagator
        GS[GS < THRESHOLD] = 0
        LOCS[0, 0] = f.positions3D(GS, peak_min_distance=PMD, num_particles='None', MPP=MPP)
        A = LOCS[0, 0].astype('int')
        LOCS[0, 1] = IM[A[:, 0], A[:, 1], A[:, 2]]
        LOCS[0, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]
            
        X.append(LOCS[0, 0][:, 0]*(1/FS))
        Y.append(LOCS[0, 0][:, 1]*(1/FS))
        Z.append(LOCS[0, 0][:, 2]*SZ)
        I_FS.append(LOCS[0, 1])
        I_GS.append(LOCS[0, 2])
            
        return [X, Y, Z, I_FS, I_GS]
    
    #%% Clean tracks with Search Sphere
    def clean_tracks_search_sphere(track, rsphere):
        import numpy as np
        import pandas as pd
        import functions as f
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        
        # rsphere = 20
        frame_skip = 10
        min_size = 50
        
        x, y, z, t, fr = track['X'].values, track['Y'].values, track['Z'].values, track['TIME'].values, track['FRAME'].values
        particle_number = track['PARTICLE'].unique()[0]
        
        track_frames = track['FRAME'].values
        frames = np.unique(track_frames)
        
        reps = f.contiguous_repeats(track_frames)
        reps_first_frame = int(reps[0])
        
        init_frames = track_frames[:reps_first_frame] 
        id_init_frames = [i for i, val in enumerate(track_frames) if val==init_frames[0]]
        
        tracks = []
        d = []
        for idi in id_init_frames:
            # print(idi)
            x0, y0, z0, t0, fr0 = x[idi], y[idi], z[idi], t[idi], fr[idi] 
            tr = [(x0, y0, z0, t0, fr0, idi)]
            dd = []
            frame_skip_counter = 0
            for k in range(idi+1, len(track)):
                # print(k)
                x1, y1, z1, t1, fr1 = x[k], y[k], z[k], t[k], fr[k]
                dist = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
                dd.append(dist)
                
                if dist <= rsphere:
                    tr.append((x1, y1, z1, t1, fr1, particle_number))
                    x0, y0, z0, t0, fr0 = x1, y1, z1, t1, fr1
                    frame_skip_counter = 0                
                    
                elif dist > rsphere:
                    frame_skip_counter += 1
                    if frame_skip_counter > frame_skip:
                        break
                        print('Broke')
    
            tracks.append(tr)
            d.append(dd)
            
            
            lengths = []
            for k in range(len(tracks)):
                lengths.append(len(tracks[k]))
                
            id_max = np.where(lengths == np.max(lengths))[0][0]
            
            trr = np.array(tracks[id_max])
            
            # fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
            # ax[0].scatter(x, -y, c=t)
            # ax[1].scatter(trr[:, 0], -trr[:, 1], c= trr[:, 3])
            # ax.axis('square')
            
            
            # fig = plt.figure(2)
            # ax1 = fig.add_subplot(111, projection='3d') 
            # # ax1.scatter(trr[:, 1], trr[:, 0], trr[:, 2], c=trr[:, 3])
            # ax1.scatter(track.X, track.Y, track.Z, c=track.TIME)
            # # ax1.scatter(track.X.values[:2], track.Y.values[:2], track.Z.values[:2], c=track.TIME.values[:2])
            # pyplot.show()
            
            x_new = trr[:, 0]
            y_new = trr[:, 1]
            z_new = trr[:, 2]
            t_new = trr[:, 3]
            fr_new = trr[:, 4]
            p_new = trr[:, 5]
            
            return [x_new, y_new, z_new, t_new, fr_new, p_new]
        
    #%% Search Sphere tracking
    def search_sphere_tracking(DF, rsphere, frame_skip, min_size):
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        
        # rsphere = 5
        # frame_skip = 10
        # min_size = 20
    
        # dd = DF[DF['FRAME'] == 0]
        # frames = np.unique(DF['FRAME'])
    
        dd = DF
        frames = np.unique(dd['FRAME'])
        num_particles = len(dd[dd['FRAME'] == 0])
    
    
    
        tracks = []
        for n in tqdm(range(num_particles)):
    
            # n = 70
            x0, y0, z0, t0, fr0 = dd['X'][n], dd['Y'][n], dd['Z'][n], dd['TIME'][n], dd['FRAME'][n]
            
            track = [(x0, y0, z0, t0, fr0, n)]
            frame_skip_counter = 0
            
            for k in range(1, len(frames)):
                s = DF[DF['FRAME'] == k]
                x, y, z, tt, fr = s['X'].values, s['Y'].values, s['Z'].values, s['TIME'].values, s['FRAME'].values
                
                dist = np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)
                
                if (dist<rsphere).any():
                    id_min = np.where(dist == dist.min())[0][0]
                    track.append((x[id_min], y[id_min], z[id_min], tt[id_min], fr[id_min], n))
                    x0, y0, z0 = x[id_min], y[id_min], z[id_min]
                    frame_skip_counter = 0
                else:
                    frame_skip_counter += 1
                    if frame_skip_counter > frame_skip:
                        break
            
            if len(track) > min_size:
                track = pd.DataFrame(np.array(track), columns=['X', 'Y', 'Z', 'TIME', 'FRAME', 'PARTICLE'])
                tracks.append(track)
                
    
        LINKED = pd.concat(tracks)  
        
        return LINKED








