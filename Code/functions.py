import cv2
import numpy as np
# %% rgb2gray
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


# %% square_image
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


# %% bandpassFilter
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

    LCO = np.exp(-((ii - MIS / 2) ** 2 + (jj - MIS / 2) ** 2) * (2 * xl / MIS) ** 2)
    SCO = np.exp(-((ii - MIS / 2) ** 2 + (jj - MIS / 2) ** 2) * (2 * xs / MIS) ** 2)
    BP = SCO - LCO
    BPP = np.fft.ifftshift(BP)

    # Filter image
    filtered = BP * img_fft
    img_filt = np.fft.ifftshift(filtered)
    img_filt = np.fft.ifft2(img_filt)
    # img_filt = np.rot90(np.real(img_filt),2)

    return img_filt, BPP

# %% videoImport
'''def videoImport(video, N):
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
'''

def videoImport(video, N):
    """
    Import video frames as a 3D array of grayscale images.

    Parameters:
    ----------
    video : str
        Path to the video file.
    N : int
        Frame number to import (N > 0 imports a specific frame).
        If N = 0, imports all frames.

    Returns:
    -------
    imStack : np.ndarray
        - If N == 0: A 3D numpy array of shape (NUM_FRAMES, HEIGHT, WIDTH) where each frame is stored as a 2D grayscale image.
        - If N > 0: A 2D numpy array of the specific frame (HEIGHT, WIDTH).

    Raises:
    ------
    ValueError:
        If the video file cannot be opened or N is out of bounds.
    """

    # Open the video file
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video}")

    # Get video properties
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Error handling: Ensure frame number is valid
    if N > num_frames:
        cap.release()
        raise ValueError(f"Frame number {N} exceeds total frames in video ({num_frames}).")

    # Import all frames if N == 0
    if N == 0:
        # Preallocate 3D array for grayscale frames (8-bit)
        imStack = np.empty((num_frames, height, width), dtype='uint8')
        success = True
        frame_idx = 0

        while success and frame_idx < num_frames:
            success, frame = cap.read()
            if success:
                # Convert frame to grayscale and store it in the stack
                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                imStack[frame_idx] = grayscale_frame
                frame_idx += 1

    # Import a specific frame if N > 0
    else:
        # Seek to the specific frame and read it
        cap.set(cv2.CAP_PROP_POS_FRAMES, N - 1)  # N-1 because frame count starts from 0
        success, frame = cap.read()
        if not success:
            cap.release()
            raise ValueError(f"Error reading frame {N} from video.")

        # Convert the frame to grayscale
        imStack = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Release the video capture object
    cap.release()

    return imStack

# %% exportAVI
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
        frame = np.uint8(255 * frame / frame.max())
        #    frame = np.random.randint(0, 255, (NI, NJ,3)).astype('uint8')
        VIDEO.write(frame)

    VIDEO.release()

    print(filename, 'exported successfully')
    return

# %% rayleighSommerfeldPropagator
def rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, bandpass, med_filter, bp_smallest_px=4,
                                 bp_largest_px=60):
    ## Rayleigh-Sommerfeld Back Propagator
    #   Inputs:          I - hologram (grayscale)
    #             I_MEDIAN - median image
    #                    Z - numpy array defining defocusing distances
    #   Output:        IMM - 3D array representing stack of images at different Z
    import numpy as np
    from scipy.ndimage import median_filter

    # Divide by Median image
    I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)
    IN = I / I_MEDIAN

    if med_filter:
        IN = median_filter(IN, size=1)

    if bandpass:
        _, BP = bandpassFilter(IN, bp_smallest_px, bp_largest_px)
        E = np.fft.fftshift(BP) * np.fft.fftshift(np.fft.fft2(IN - 1))
    else:
        E = np.fft.fftshift(np.fft.fft2(IN - 1))

    # Patameters     #Set as input parameters
    # N = 1.3226               # Index of refraction
    LAMBDA = LAMBDA  # HeNe
    FS = FS  # Sampling Frequency px/um
    NI = np.shape(IN)[0]  # Number of rows
    NJ = np.shape(IN)[1]  # Nymber of columns
    # SZ = 10
    Z = SZ * np.arange(0, NUMSTEPS)
    # Z = (FS * (51 / 31)) * np.arange(0, NUMSTEPS)
    #    Z = SZ*np.arange(0, NUMSTEPS)
    K = 2 * np.pi * N / LAMBDA  # Wavenumber

    # Rayleigh-Sommerfeld Arrays
    # P = np.empty_like(I_MEDIAN, dtype='complex64')

    # for i in range(NI):
    #    for j in range(NJ):
    #        P[i, j] = ((LAMBDA * FS) / (max([NI, NJ]) * N)) ** 2 * ((i - NI / 2) ** 2 + (j - NJ / 2) ** 2)

    jj, ii = np.meshgrid(np.arange(NJ), np.arange(NI))
    const = ((LAMBDA * FS) / (max([NI, NJ]) * N)) ** 2
    P = const * ((ii - NI / 2) ** 2 + (jj - NJ / 2) ** 2)

    if (P > 1).any():
        P = P / P.max()

    P = np.conj(P)
    Q = np.sqrt(1 - P) - 1

    if all(Z >= 0):
        Q = np.conj(Q)

    # R = np.empty([NI, NJ, Z.shape[0]], dtype=complex)
    IZ = np.empty([NI, NJ, Z.shape[0]], dtype='float32')

    for k in range(Z.shape[0]):
        R = np.exp((-1j * K * Z[k] * Q), dtype='complex64')
        IZ[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E * R)))
    #        print(('RS', k))
    return IZ


# %% medianImage
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


# %% zGradientStack
def zGradientStack(IM):
    # Z-Gradient Stack
    #   Inputs:   I - hologram (grayscale)
    #            IM - median image
    #             Z - numpy array defining defocusing distances
    #   Output: CONV - 3D array representing stack of images at different Z
    import numpy as np
    from scipy import ndimage

    #    I = mpimg.imread('131118-1.png')
    #    I_MEDIAN = mpimg.imread('AVG_131118-2.png')
    #    Z = 0.02*np.arange(1, 151)
    #     IM = rayleighSommerfeldPropagator(I, I_MEDIAN, Z)

    # % Sobel-type kernel
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


# %% modified_propagator
def modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, bandpass, med_filter):
    ## Modified Propagator
    #   Inputs:          I - hologram (grayscale)
    #             I_MEDIAN - median image
    #                    Z - numpy array defining defocusing distances
    #   Output:        GS - 3D Gradient Stack

    import numpy as np
    from functions import bandpassFilter
    from scipy.ndimage import median_filter

    # Divide by Median image
    I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)
    IN = I / I_MEDIAN

    if med_filter:
        IN = median_filter(IN, size=1)

    # Bandpass Filter
    if bandpass:
        _, BP = bandpassFilter(IN, 2, 30)
        E = np.fft.fftshift(BP) * np.fft.fftshift(np.fft.fft2(IN - 1))
    else:
        E = np.fft.fftshift(np.fft.fft2(IN - 1))

    # Patameter
    LAMBDA = LAMBDA  # HeNe
    FS = FS  # Sampling Frequency px/um
    NI = np.shape(IN)[0]  # Number of rows
    NJ = np.shape(IN)[1]  # Nymber of columns
    Z = SZ * np.arange(0, NUMSTEPS)
    K = 2 * np.pi * N / LAMBDA  # Wavenumber

    # Rayleigh-Sommerfeld Arrays
    jj, ii = np.meshgrid(np.arange(NJ), np.arange(NI))
    const = ((LAMBDA * FS) / (max([NI, NJ]) * N)) ** 2
    q = (ii - NI / 2) ** 2 + (jj - NJ / 2) ** 2

    # const = ((LAMBDA*FS)/(max([NI, NJ])*N))**2
    # ff = np.fft.fftfreq(NI, FS)
    # ff = ff**2+ff**2
    # P = const*ff

    P = const * q

    if (P > 1).any():
        P = P / P.max()

    P = np.conj(P)
    Q = np.sqrt(1 - P) - 1

    if all(Z > 0):
        Q = np.conj(Q)

    GS = np.empty([NI, NJ, Z.shape[0]], dtype='float32')

    for k in range(Z.shape[0]):
        R = 2 * np.pi * 1j * q * np.exp(1j * K * Z[k] * Q)
        # GS[:, :, k] = np.abs(1 + np.fft.ifft2(np.fft.ifftshift(E*R)))
        GS[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E * R)))

    # _, BINS = np.histogram(GS.flatten(), bins=100)
    # GS[GS < BINS[60]] = 0   # 60
    # GS[GS < 400] = 0

    return GS


def bgPathToArray(image_path):
    """
    Converts a PNG image to a 2D NumPy array in 8-bit unsigned integer format (U8),
    similar to LabVIEW's IMAQ ImageToArray VI behavior.

    Parameters:
    ----------
    image_path : str
        Path to the PNG image file.

    Returns:
    -------
    np.ndarray
        A 2D NumPy array representing the grayscale image (Height x Width) in U8 format.

    Raises:
    ------
    ValueError:
        If the image cannot be loaded from the provided path.
    """
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Directly loads as a 2D array (grayscale)

    # Error handling if the image is not found or cannot be opened
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")

    # Ensure the image is in 8-bit unsigned integer format (U8)
    image_u8 = image.astype(np.uint8)

    return image_u8


# %% Positions batch
def positions_batch(TUPLE):
    import numpy as np
    import functions as f
    I = TUPLE[0]
    I_MEDIAN = TUPLE[1]
    N = TUPLE[2]
    LAMBDA = TUPLE[3]
    MPP = TUPLE[4]
    FS = (MPP / 10) * 0.711
    SZ = TUPLE[5]
    NUMSTEPS = TUPLE[6]
    THRESHOLD = TUPLE[7]
    PMD = TUPLE[8]

    LOCS = np.empty((1, 3), dtype=object)
    X, Y, Z, I_FS, I_GS = [], [], [], [], []
    IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, True, False).astype('float32')
    GS = f.zGradientStack(IM).astype('float32')
    # GS = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS)  # Modified propagator
    GS[GS < THRESHOLD] = 0
    LOCS[0, 0] = f.positions3D(GS, peak_min_distance=PMD, num_particles='None', MPP=MPP)
    A = LOCS[0, 0].astype('int')
    LOCS[0, 1] = IM[A[:, 0], A[:, 1], A[:, 2]]
    LOCS[0, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]

    X.append(LOCS[0, 0][:, 0] * (1 / FS))
    Y.append(LOCS[0, 0][:, 1] * (1 / FS))
    Z.append(LOCS[0, 0][:, 2] * SZ)
    I_FS.append(LOCS[0, 1])
    I_GS.append(LOCS[0, 2])

    return [X, Y, Z, I_FS, I_GS]


# %% Positions batch modified
def positions_batch_modified(TUPLE):
    import numpy as np
    import functions as f
    I = TUPLE[0]
    I_MEDIAN = TUPLE[1]
    N = TUPLE[2]
    LAMBDA = TUPLE[3]
    MPP = TUPLE[4]
    FS = (MPP / 10) * 0.711
    SZ = TUPLE[5]
    NUMSTEPS = TUPLE[6]
    THRESHOLD = TUPLE[7]
    PMD = TUPLE[8]

    # 0   1    2  3    4    5    6         7
    # zip(IT, MED, n, lam, mpp, sz, numsteps, pmd)

    LOCS = np.empty((1, 3), dtype=object)
    X, Y, Z, I_FS, I_GS = [], [], [], [], []
    # IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, True, False).astype('float32')
    # GS = f.zGradientStack(IM).astype('float32')  
    GS = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, True, True)  # Modified propagator
    GS[GS < THRESHOLD] = 0
    LOCS[0, 0] = f.positions3D(GS, peak_min_distance=PMD, num_particles='None', MPP=MPP)
    A = LOCS[0, 0].astype('int')
    LOCS[0, 1] = GS[A[:, 0], A[:, 1], A[:, 2]]
    LOCS[0, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]

    X.append(LOCS[0, 0][:, 0] * (1 / FS))
    Y.append(LOCS[0, 0][:, 1] * (1 / FS))
    Z.append(LOCS[0, 0][:, 2] * SZ)
    I_FS.append(LOCS[0, 1])
    I_GS.append(LOCS[0, 2])

    return [X, Y, Z, I_FS, I_GS]

# %% Positions3D
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

    D1 = int(MPP / 10)
    D2 = int(MPP / 10)
    Z_SUM_XY = np.empty((GS.shape[2], len(PKS)))
    for ii in range(len(PKS)):
        idi = PKS[ii, 0]
        idj = PKS[ii, 1]
        A = GS[idi - D1:idi + D2:, idj - D1:idj + D2, :]  # How to treat borders?
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
    w = 2  # 2
    pol = lambda a, x: a[0] * x ** 2 + a[1] * x + a[2]
    z_max = []

    for j in range(len(Z_SUM_XY_MAXS)):
        i = Z_SUM_XY_MAXS[j][0]
        idi = np.arange(i - w, i + w + 1)
        temp = np.pad(Z_SUM_XY, ((w, w), (0, 0)))
        val = temp[idi + w, j]

        coefs = np.polyfit(idi, val, 2)

        interp_idi = np.linspace(idi[0], idi[-1], 10)
        interp_val = pol(coefs, interp_idi)

        idi_max = np.where(interp_val == interp_val.max())[0][0]
        z_max.append(interp_idi[idi_max])

    # XYZ_POSITIONS = np.hstack((XYZ_POSITIONS, Z_SUM_XY_MAXS[:, 0]))    # YXZ_POSITIONS = np.insert(PKS, 2, Z_SUM_XY_MAXS[:, 0], axis=-1)         # Actually [Y, X, Z]
    YXZ_POSITIONS = np.insert(np.float16(PKS), 2, z_max, axis=-1)

    return YXZ_POSITIONS  # (x,y) in pixels, z in slice number