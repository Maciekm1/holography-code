import numpy as np
import matplotlib.pyplot as plt
import cv2


zp = [i/5 for i in range(30)]      # um


def simulate_pattern(zp):
    
    lam = 0.505    # um
    a = 0.1        # um
    Np = 1.55
    Nm = 1.33
    FS = 10        # um

    
    k = 2*np.pi*Nm/lam
    xy = np.arange(-256, 257) / FS
    x, y = np.meshgrid(xy, xy)
    
    R = (x**2 + y**2)**(1/2)
    theta = np.arctan(R/zp)
    
    r = (x**2+y**2+zp**2)**(1/2)
    V = 4*np.pi*a**3 / 3
    q = 2*k*np.sin(theta/2)
    f = 3*(q*a)**(-3) * (np.sin(q*a) - q*a*np.cos(q*a))
    c = (k**2 * np.exp(1j*k*r)/(2*np.pi*r)) * (Nm-1) * V * f * (1 + np.cos(theta))
    b = 1 + np.real(c)
    
    b = np.nanmax(b) - b
    b = 255*b / np.nanmax(b)
    b = np.uint8(b) 
    
    mask = np.zeros_like(b)
    mask[256, 256] = 1
    
    return cv2.inpaint(b, mask, 3, flags = cv2.INPAINT_NS)

b = list(map(simulate_pattern, zp))


for k in range(len(b)):
    plt.imshow(b[k], cmap='gray')
    plt.title('Distance from hologram plane is {}'.format(zp[k]))
    plt.show()
    plt.pause(0.5)