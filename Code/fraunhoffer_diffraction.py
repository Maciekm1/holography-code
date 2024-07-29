#%%
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functions import imshow_sequence

def fraunhoffer(xy_mesh, W, z):
    (x, y) = xy_mesh
    LAMBDA = 642
    # z = 0.004
    return (1-np.pi*sp.j1((np.pi*W*np.sqrt((x-len(x)/2)**2+(y-len(y)/2)**2)/LAMBDA)/z)/((np.pi*W*np.sqrt((x-len(x)/2)**2+(y-len(y)/2)**2)/LAMBDA)/z))**2
            

#%% W fit
x_range = np.arange(1, 100)
y_range = np.arange(1, 100)
xy = np.meshgrid(x_range, y_range)
W = 1
z = 0.01

f = fraunhoffer(xy, W, z)
f_noise = f + 0.2*np.random.randn(len(x_range), len(y_range))

w_fit = np.linspace(0.1, W + 1, 200)
chisq = np.empty_like(w_fit)

for i in range(len(w_fit)):
    chisq[i] = np.sum((f_noise - fraunhoffer(xy, w_fit[i], z))**2) / np.var(f_noise)

w_min = w_fit[chisq == chisq.min()] 
f_fit = fraunhoffer(xy, w_min, z)


plt.figure(figsize=(14, 9))
plt.subplot(2, 2, 3)
plt.plot(w_fit, chisq, '.-', label=r'$\chi^{2}$')
plt.plot(w_min, chisq[w_fit == w_min], 'ro', label=r'$w_{min}$')
plt.xlabel('w')
plt.ylabel(r'$\chi^{2}$')
plt.legend()
plt.grid()

plt.subplot(2, 2, 1)
plt.imshow(f, cmap='gray')
plt.title(r'$w_{real}$ = '+np.str(W))
plt.subplot(2, 2, 2)
plt.imshow(f_noise, cmap='gray')
plt.title('With random noise')
plt.subplot(2, 2, 4)
plt.imshow(f_fit, cmap='gray')
plt.title(r'$w_{fitted}$ = '+np.str(w_min[0].astype('float16')))

plt.show()

#%% W - z fit
x_range = np.arange(1, 50)
y_range = np.arange(1, 50)
xy = np.meshgrid(x_range, y_range)
W = 7
z = 0.01

f = fraunhoffer(xy, W, z)
f_noise = f + 0.05*np.random.randn(len(x_range), len(y_range))

w_fit = np.linspace(0.1, W + 2, 20)
z_fit = np.linspace(0.01, z+0.01, 20)
chisq = np.empty((len(w_fit), len(z_fit)))

for i in range(len(w_fit)):
    for j in range(len(z_fit)):
        chisq[i, j] = np.sum((f_noise - fraunhoffer(xy, w_fit[i], z_fit[j]))**2) / np.var(f_noise)

i_min, j_min = np.where(chisq == chisq.min())
w_min = w_fit[i_min] 
z_min = z_fit[j_min]
f_fit = fraunhoffer(xy, w_min, z_min)


fig = plt.figure(figsize=(14, 9))
# ax3 = fig.add_subplot(223, projection='3d')
# # ax = Axes3D(fig)

# xy_chisq = np.meshgrid(z_fit, w_fit)
# ax3.plot_surface(xy_chisq[1], xy_chisq[0], np.log(chisq), cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax3.scatter(w_min, z_min, np.log(chisq[i_min, j_min]), c='r')
# ax3.set_xlabel('w')
# ax3.set_ylabel('z')
# ax3.set_zlabel(r'$\chi^{2}$')

X, Y = np.meshgrid(z_fit, w_fit)
ax3 = fig.add_subplot(223)
ax3.contourf(X, Y, chisq, cmap='viridis')
ax3.scatter(z_min, w_min, c='red')
ax3.set_xlabel('z')
ax3.set_ylabel('w')
ax3.set_title(r'$\chi^{2}$')


ax1 = fig.add_subplot(221)
ax1.imshow(f, cmap='gray', interpolation='bilinear')
ax1.set_title(r'$w_{real}$ = '+np.str(W)+r', $z_{real}$ ='+np.str(z))

ax2 = fig.add_subplot(222)
ax2.imshow(f_noise, cmap='gray', interpolation='bilinear')
ax2.set_title('With random noise')

ax4 = fig.add_subplot(224)
ax4.imshow(f_fit, cmap='gray', interpolation='bilinear')
ax4.set_title(r'$w_{fitted}$ = '+np.str(w_min[0].astype('float16'))+r', $z_{fitted}$ = '+np.str(z_min[0].astype('float16')))

plt.show()

#%%
import easygui as gui
import matplotlib.image as mpimg
import pandas as pd

path = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/', multiple=True)

I = mpimg.imread(path[0])
IM = mpimg.imread(path[2])
POSITIONS = pd.read_csv(path[1], index_col=0)
IN = I / IM

#%%
positions_frame1 = POSITIONS[POSITIONS.FRAME == 0].values

# plt.figure()
# plt.imshow(IN, cmap='gray')
# plt.scatter(positions_frame1[0, 1], positions_frame1[0, 0], c='r')
# plt.show()

padd = 19
f_noise = IN[int(positions_frame1[0, 0])-padd-1:int(positions_frame1[0, 0])+padd, int(positions_frame1[0, 1])-padd-1:int(positions_frame1[0, 1])+padd]
# f_noise = f_noise/f_noise.max()
# plt.figure()
# plt.imshow(f)
# plt.show()
shape = np.shape(f_noise)

x_range = np.arange(1, shape[0]+1)
y_range = np.arange(1, shape[1]+1)
xy = np.meshgrid(x_range, y_range)
W  = 2
z = positions_frame1[0, 2]

w_fit = np.linspace(0.1, 20, 20)
chisq = np.empty_like(w_fit)

for i in range(len(w_fit)):
    chisq[i] = np.sum((f_noise - fraunhoffer(xy, w_fit[i], z))**2) / np.var(f_noise)

w_min = w_fit[chisq == chisq.min()] 
f_fit = fraunhoffer(xy, w_min, z)

