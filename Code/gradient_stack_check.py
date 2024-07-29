import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functions as f
import easygui as gui
import ipywidgets as widgets
from skimage.feature import peak_local_max
from scipy import ndimage
#get_ipython().run_line_magic('matplotlib', 'qt')
#%matplotlib widget

files = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/', multiple=True)

path = [[],[]]
if files[0].find('MED') != -1:
    path[0] = files[1]
    path[1] = files[0]
else:
    path = files


I =mpimg.imread(path[0]) 
I_MEDIAN = mpimg.imread(path[1])
#I_MEDIAN = np.ones((np.shape(I)[0], np.shape(I)[1]))

N = 1.3226
LAMBDA = 0.642               # Diode
MPP = 20                      # Magnification: 10x, 20x, 50x, etc
FS = 0.711*(MPP/10)                     # Sampling Frequency px/um
NI = np.shape(I)[0]
NJ = np.shape(I)[1]
SZ = 2.5                       # Step size in um
# Z = (FS*(51/31))*np.arange(0, 150)       # Number of steps

# %%
# Z = SZ*np.arange(0, 150)
# ZZ = np.linspace(0, SZ*149, 150)
# Z = FS*ZZ
# K = 2*np.pi*N/LAMBDA            # Wavenumber
NUMSTEPS = 50

RS = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, True, True)

# Sobel-type kernel
SZ0 = np.array(([-1, -2, -1], [-2, -4, -2], [-1, -2, -1]))
SZ1 = np.zeros_like(SZ0)
SZ2 = -SZ0
SZ = np.stack((SZ0, SZ1, SZ2), axis=-1)

# Convolution IM*SZ
IMM = np.dstack((RS[:, :, 0][:, :, np.newaxis], RS, RS[:, :, -1][:, :, np.newaxis]))
GS = ndimage.convolve(RS, SZ, mode='mirror')
GS = np.delete(GS, [0, np.shape(GS)[2]-1], axis=2)

# set up plot
fig, ax = plt.subplots(figsize=(6, 4)) 
 
 
@widgets.interact(threshold=(0, 0.5, 0.05), peak_min_dist=(0, 50, 5), show_scatter = False)
def update(threshold=0.1, peak_min_dist=30, show_scatter=True):
    GSS = np.copy(GS)
    GSS[GS < threshold] = 0
    ZP = np.max(GSS, axis=-1)
    PKS = peak_local_max(ZP, min_distance=peak_min_dist)
    ax.clear()
    ax.imshow(ZP, cmap='gray')
    if show_scatter==True:
        ax.scatter(PKS[:,1], PKS[:,0], marker='o', facecolors='none', s=80, edgecolors='r')
