import cv2
import numpy as np
from skimage.feature import peak_local_max
import os
from cv2 import VideoWriter, VideoWriter_fourcc
from scipy import ndimage
from scipy.ndimage import median_filter
import sys

# Temp Constant. Uses Median for BG if set to true, uses bg_image if set to false
USE_MEDIAN = False

def bandpassFilter(img, xs, xl):
    """
    Apply a bandpass filter to a grayscale image.

    Parameters:
    ----------
    img : np.ndarray
        Grayscale image array (2D).
    xs : float
        Small cutoff size (Pixels).
    xl : float
        Large cutoff size (Pixels).

    Returns:
    -------
    img_filt : np.ndarray
        Filtered image.
    BPP : np.ndarray
        Bandpass filter array.
    """
    img_fft = np.fft.fftshift(np.fft.fft2(img))
    img_amp = abs(img_fft)

    ni, nj = img_amp.shape
    MIS = ni

    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni))

    LCO = np.exp(-((ii - MIS / 2) ** 2 + (jj - MIS / 2) ** 2) * (2 * xl / MIS) ** 2)
    SCO = np.exp(-((ii - MIS / 2) ** 2 + (jj - MIS / 2) ** 2) * (2 * xs / MIS) ** 2)
    BP = SCO - LCO
    BPP = np.fft.ifftshift(BP)

    # Filter image
    filtered = BP * img_fft
    img_filt = np.fft.ifftshift(filtered)
    img_filt = np.fft.ifft2(img_filt)

    return img_filt, BPP


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


def rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, bandpass, med_filter,
                                    bp_smallest_px=4, bp_largest_px=60):
    """
    Rayleigh-Sommerfeld Back Propagator.

    Parameters:
    ----------
    I : np.ndarray
        Hologram (grayscale).
    I_MEDIAN : np.ndarray
        Median image.
    N : float
        Index of refraction.
    LAMBDA : float
        Wavelength.
    FS : float
        Sampling Frequency px/um.
    SZ : float
        Step size.
    NUMSTEPS : int
        Number of steps in propagation.
    bandpass : bool
        Whether to apply bandpass filtering.
    med_filter : bool
        Whether to apply median filtering.

    Returns:
    -------
    IZ : np.ndarray
        3D array representing stack of images at different Z.
    """
    if USE_MEDIAN:
        I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)
        IN = I / I_MEDIAN
        #print(I_MEDIAN.shape)
    else:
        bg_array = bgPathToArray('C:/Users/mz1794/Downloads/Python and Viking DHM port-20241010T094616Z-001/Python and Viking DHM port/1024bg.png')
        if bg_array.shape[0] != I.shape[0]:
            #bg_array = np.mean(bg_array, axis=0, keepdims=True)  # Adjust bg_array to match I's shape using mean
            bg_array = bg_array[:50, :]  # Now shape is (50, 1024)
        IN = I / bg_array

    # Correct up to this point, Corresponds to  I/ bg_array division in LabVIEW, not sure about the decrement after(in labVIEW vs this)?
    #print('*'*150)
    #print(I)
    #print(I.shape)
    #print('-' * 150)
    #print(IN)
    #print(IN.shape)
    #print('*'*150)
    #sys.exit()

    if med_filter:
        IN = median_filter(IN, size=1)

    if bandpass:
        _, BP = bandpassFilter(IN, bp_smallest_px, bp_largest_px)
        E = np.fft.fftshift(BP) * np.fft.fftshift(np.fft.fft2(IN - 1))
    else:
        E = np.fft.fftshift(np.fft.fft2(IN - 1))

    # TEST IF BANDPASS CORRECT
    print('*'*150)
    print(E)
    print(E.shape)
    print('*'*150)
    sys.exit()

    LAMBDA = LAMBDA
    FS = FS
    NI, NJ = IN.shape
    Z = SZ * np.arange(0, NUMSTEPS)
    K = 2 * np.pi * N / LAMBDA  # Wavenumber

    jj, ii = np.meshgrid(np.arange(NJ), np.arange(NI))
    const = ((LAMBDA * FS) / (max(NI, NJ) * N)) ** 2
    P = const * ((ii - NI / 2) ** 2 + (jj - NJ / 2) ** 2)

    if (P > 1).any():
        P = P / P.max()

    P = np.conj(P)
    Q = np.sqrt(1 - P) - 1

    if all(Z >= 0):
        Q = np.conj(Q)

    IZ = np.empty([NI, NJ, Z.shape[0]], dtype='float32')

    for k in range(Z.shape[0]):
        R = np.exp((-1j * K * Z[k] * Q), dtype='complex64')
        IZ[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E * R)))

    return IZ

def medianImage(VID, num_frames):
    """
    Calculate the median image from a 3D numpy array of video frames.

    Parameters:
    ----------
    VID : np.ndarray
        3D numpy array of video file.
    num_frames : int
        Number of frames to calculate the median image.

    Returns:
    -------
    MEAN : np.ndarray
        2D pixel mean array.
    """
    def spaced_elements(array, num_elems):
        return array[np.round(np.linspace(0, len(array) - 1, num_elems)).astype(int)]

    N = VID.shape[2]
    id = spaced_elements(np.arange(N), num_frames)
    MEAN = np.median(VID[:, :, id], axis=2)

    return MEAN


def zGradientStack(IM):
    """
    Calculate the Z-gradient stack of an image.

    Parameters:
    ----------
    IM : np.ndarray
        Input image array.

    Returns:
    -------
    GS : np.ndarray
        3D array representing the gradient stack.
    """
    SZ0 = np.array([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], dtype='float')
    SZ1 = np.zeros_like(SZ0)
    SZ2 = -SZ0
    SZ = np.stack((SZ0, SZ1, SZ2), axis=2)

    IMM = np.dstack((IM[:, :, 0][:, :, np.newaxis], IM, IM[:, :, -1][:, :, np.newaxis]))
    GS = ndimage.convolve(IMM, SZ, mode='mirror')
    GS = np.delete(GS, [0, GS.shape[2] - 1], axis=2)

    return GS


def modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, bandpass, med_filter):
    """
    Modified Propagator.

    Parameters:
    ----------
    I : np.ndarray
        Hologram (grayscale).
    I_MEDIAN : np.ndarray
        Median image.
    N : float
        Index of refraction.
    LAMBDA : float
        Wavelength.
    FS : float
        Sampling Frequency in px/um.
    SZ : float
        Step size.
    NUMSTEPS : int
        Number of steps in propagation.
    bandpass : bool
        Whether to apply bandpass filtering.
    med_filter : bool
        Whether to apply median filtering.

    Returns:
    -------
    GS : np.ndarray
        3D Gradient Stack.
    """
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

    # Parameters
    NI, NJ = IN.shape  # Number of rows and columns
    Z = SZ * np.arange(NUMSTEPS)
    K = 2 * np.pi * N / LAMBDA  # Wavenumber

    # Rayleigh-Sommerfeld Arrays
    jj, ii = np.meshgrid(np.arange(NJ), np.arange(NI))
    const = ((LAMBDA * FS) / (max(NI, NJ) * N)) ** 2
    q = (ii - NI / 2) ** 2 + (jj - NJ / 2) ** 2

    P = const * q

    if (P > 1).any():
        P /= P.max()

    P = np.conj(P)
    Q = np.sqrt(1 - P) - 1

    if all(Z > 0):
        Q = np.conj(Q)

    GS = np.empty((NI, NJ, Z.shape[0]), dtype='float32')

    for k in range(Z.shape[0]):
        R = 2 * np.pi * 1j * q * np.exp(1j * K * Z[k] * Q)
        GS[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E * R)))

    return GS


def positions_batch(TUPLE):
    """
    Process a batch of positions from the input tuple.

    Parameters:
    ----------
    TUPLE : tuple
        A tuple containing the following elements:
        - I (np.ndarray): Hologram.
        - I_MEDIAN (np.ndarray): Median image.
        - N (float): Index of refraction.
        - LAMBDA (float): Wavelength.
        - MPP (float): Microns per pixel.
        - SZ (float): Step size.
        - NUMSTEPS (int): Number of steps in propagation.
        - THRESHOLD (float): Threshold for GS.
        - PMD (float): Peak minimum distance.

    Returns:
    -------
    list
        A list containing the X, Y, Z positions and intensity arrays (I_FS, I_GS).
    """
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

    IM = rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, True, False).astype('float32')
    GS = zGradientStack(IM).astype('float32')

    GS[GS < THRESHOLD] = 0
    LOCS[0, 0] = positions3D(GS, peak_min_distance=PMD, num_particles='None', MPP=MPP)
    A = LOCS[0, 0].astype('int')

    LOCS[0, 1] = IM[A[:, 0], A[:, 1], A[:, 2]]
    LOCS[0, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]

    X.append(LOCS[0, 0][:, 0] * (1 / FS))
    Y.append(LOCS[0, 0][:, 1] * (1 / FS))
    Z.append(LOCS[0, 0][:, 2] * SZ)
    I_FS.append(LOCS[0, 1])
    I_GS.append(LOCS[0, 2])

    return [X, Y, Z, I_FS, I_GS]


def positions_batch_modified(TUPLE):
    """
    Process a modified batch of positions from the input tuple.

    Parameters:
    ----------
    TUPLE : tuple
        A tuple containing the following elements:
        - I (np.ndarray): Hologram.
        - I_MEDIAN (np.ndarray): Median image.
        - N (float): Index of refraction.
        - LAMBDA (float): Wavelength.
        - MPP (float): Microns per pixel.
        - SZ (float): Size parameter.
        - NUMSTEPS (int): Number of steps in propagation.
        - THRESHOLD (float): Threshold for GS.
        - PMD (float): Peak minimum distance.

    Returns:
    -------
    list
        A list containing the X, Y, Z positions and intensity arrays (I_FS, I_GS).
    """
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

    GS = modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, True, True)
    GS[GS < THRESHOLD] = 0

    LOCS[0, 0] = positions3D(GS, peak_min_distance=PMD, num_particles='None', MPP=MPP)
    A = LOCS[0, 0].astype('int')

    LOCS[0, 1] = GS[A[:, 0], A[:, 1], A[:, 2]]
    LOCS[0, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]

    X.append(LOCS[0, 0][:, 0] * (1 / FS))
    Y.append(LOCS[0, 0][:, 1] * (1 / FS))
    Z.append(LOCS[0, 0][:, 2] * SZ)
    I_FS.append(LOCS[0, 1])
    I_GS.append(LOCS[0, 2])

    return [X, Y, Z, I_FS, I_GS]


def positions3D(GS, peak_min_distance, num_particles, MPP):
    """
    Find the 3D positions of particles in the gradient stack.

    Parameters:
    ----------
    GS : np.ndarray
        3D Gradient Stack.
    peak_min_distance : float
        Minimum distance between peaks.
    num_particles : int or str
        Number of particles to detect; 'None' for no limit.
    MPP : float
        Microns per pixel.

    Returns:
    -------
    np.ndarray
        2D array with particle positions (x, y in pixels, z in slice number).
    """
    ZP = np.max(GS, axis=-1)

    if num_particles == 'None':
        PKS = peak_local_max(ZP, min_distance=peak_min_distance)
    elif num_particles > 0:
        PKS = peak_local_max(ZP, min_distance=peak_min_distance, num_peaks=num_particles)

    D1 = int(MPP / 10)
    D2 = int(MPP / 10)
    Z_SUM_XY = np.empty((GS.shape[2], len(PKS)))

    for ii in range(len(PKS)):
        idi, idj = PKS[ii]
        A = GS[idi - D1:idi + D2, idj - D1:idj + D2, :]
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
    w = 2
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

    YXZ_POSITIONS = np.insert(np.float16(PKS), 2, z_max, axis=-1)

    return YXZ_POSITIONS  # (x,y) in pixels, z in slice number


def exportAvi(filename, IM, NI, NJ, fps):
    """
    Export a 3D array to a .AVI movie file.

    Parameters:
    ----------
    filename : str
        Name of the output .AVI file.
    IM : np.ndarray
        3D numpy array containing image data.
    NI : int
        Number of rows of the array.
    NJ : int
        Number of columns of the array.
    fps : int
        Frames per second of output file.
    """
    dir = os.getcwd()
    file_path = os.path.join(dir, filename)
    fourcc = VideoWriter_fourcc(*'MJPG')
    video = VideoWriter(file_path, fourcc, float(fps), (NJ, NI), 0)

    for i in range(IM.shape[2]):
        frame = IM[:, :, i]
        frame = np.uint8(255 * frame / frame.max())
        video.write(frame)

    video.release()
    print(f"{filename} exported successfully")
