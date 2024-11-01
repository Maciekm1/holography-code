import cv2
import numpy as np
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.ndimage import median_filter


def bandpass_filter(img, xs, xl):
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

    # BP matches up with LabVIEW
    # TODO: I changed this from SCO - LCO to match up with LabVIEW - probably wrong?
    BP = LCO - SCO

    # Applies fft shift to normalize BP? Does not match up with LabVIEW anymore
    BPP = np.fft.ifftshift(BP)

    # Filter image - Edit: This is not used anywhere?
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


def rayleigh_sommerfeld_propagator(
        hologram, median_image, refractive_index, wavelength, start_refocus_value, sampling_freq,
        step_size, num_steps, apply_bandpass, apply_median_filter, bg_image, use_bg_image,
        bandpass_min_px=4, bandpass_max_px=60
):
    """
    Rayleigh-Sommerfeld Back Propagator.

    Parameters:
    ----------
    hologram : np.ndarray
        Grayscale hologram image.
    median_image : np.ndarray
        Median image for normalization.
    refractive_index : float
        Index of refraction.
    wavelength : float
        Wavelength in micrometers.
    start_depth : float
        Starting depth (refocus value).
    sampling_freq : float
        Sampling frequency in px/um.
    step_size : float
        Step size for depth increments.
    num_steps : int
        Number of depth steps to compute.
    apply_bandpass : bool
        Whether to apply bandpass filtering.
    apply_median_filter : bool
        Whether to apply median filtering.
    bg_image : np.ndarray
        Background image for normalization, if used.
    use_bg_image : bool
        Whether to use background image for normalization.
    bandpass_min_px : int, optional
        Minimum pixel size for bandpass filter (default is 4).
    bandpass_max_px : int, optional
        Maximum pixel size for bandpass filter (default is 60).

    Returns:
    -------
    reconstructed_stack : np.ndarray
        3D array where each slice represents the reconstructed hologram at a specific depth.
    """

    # Normalize hologram based on background or median image
    if use_bg_image:
        #TODO: Get rid of 0's in bg?
        normalized_hologram = hologram / bg_image
    else:
        # Get rid of 0's
        median_image[median_image == 0] = np.mean(median_image)
        normalized_hologram = hologram / median_image

    # Correct up to this point, Corresponds to  I/ bg_array division in LabVIEW,
    # Update: decrement + fft happens below in fft_after_decrement variable i.e. code is correct/ matches up with
    # LabVIEW up to fft_after_decrement - then it is shifted (fft.fftshift) to normalize and diverges from labVIEW code.

    # Apply median filtering if specified, currently not used, Should it be?
    if apply_median_filter:
        normalized_hologram = median_filter(normalized_hologram, size=1)

    # Compute the initial frequency spectrum of the hologram
    if apply_bandpass:
        _, _bandpass_filter = bandpass_filter(normalized_hologram, bandpass_min_px, bandpass_max_px)
        fft_after_decrement = np.fft.fft2(normalized_hologram - 1)
        spectrum = np.fft.fftshift(_bandpass_filter) * np.fft.fftshift(fft_after_decrement)
    else:
        spectrum = np.fft.fftshift(np.fft.fft2(normalized_hologram - 1))

    # Set constants for propagation
    img_height, img_width = normalized_hologram.shape

    # depth_levels is an array representing depth values, beginning from SRV (start refocus value) and incrementing by
    # step_size for num_steps steps.
    depth_levels = start_refocus_value + (step_size * np.arange(num_steps))
    wavenumber = 2 * np.pi * refractive_index / wavelength

    # Generate spatial frequency grid
    x_indices, y_indices = np.meshgrid(np.arange(img_width), np.arange(img_height))
    spatial_constant = ((wavelength * sampling_freq) / (max(img_height, img_width) * refractive_index)) ** 2

    # This effectively represents the spatial frequency squared at each point on the grid.
    # It is offset by img_height / 2 and img_width / 2 to center the zero-frequency component.
    spatial_freq_grid = spatial_constant * ((y_indices - img_height / 2) ** 2 + (x_indices - img_width / 2) ** 2)

    # Normalize spatial frequency grid
    if (spatial_freq_grid > 1).any():
        spatial_freq_grid /= spatial_freq_grid.max()
    spatial_freq_grid = np.conj(spatial_freq_grid)

    # Calculate phase shift
    # Since spatial_freq_grid was centered, phase_shift now contains phase shifts relative to the image center
    phase_shift = np.sqrt(1 - spatial_freq_grid) - 1
    if all(depth_levels >= 0):
        phase_shift = np.conj(phase_shift)

    # Initialize the output stack
    reconstructed_stack = np.empty([img_height, img_width, depth_levels.shape[0]], dtype='float32')

    # Calculate each depth slice
    for k, depth in enumerate(depth_levels):
        depth_phase_shift = np.exp(-1j * wavenumber * depth * phase_shift, dtype='complex64')

        # Inverse FFT? (np.fft.ifft2)
        reconstructed_slice = np.real(1 + np.fft.ifft2(np.fft.ifftshift(spectrum * depth_phase_shift)))
        reconstructed_stack[:, :, k] = reconstructed_slice

    return reconstructed_stack


# NOT USED IF USING BG IMAGE
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


def z_gradient_stack(image):
    """
    Calculate the Z-gradient stack of a 3D image, emphasizing intensity changes along the Z-dimension.

    Parameters:
    ----------
    image : np.ndarray
        Input 3D image array.

    Returns:
    -------
    gradient_stack : np.ndarray
        3D array representing the gradient stack along the Z-axis.
    """

    # Define 3x3 filter arrays for the Z-gradient computation, Sobel filter?
    filter_front = np.array([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], dtype='float')
    filter_center = np.zeros_like(filter_front)
    filter_back = -filter_front

    # Stack filters along the third axis to form a 3D gradient filter
    gradient_filter = np.stack((filter_front, filter_center, filter_back), axis=2)

    # Pad the image along the Z-axis by duplicating the first and last slices
    padded_image = np.dstack((image[:, :, 0][:, :, np.newaxis], image, image[:, :, -1][:, :, np.newaxis]))

    # Convolve the padded image with the gradient filter to compute Z-gradient
    gradient_stack = ndimage.convolve(padded_image, gradient_filter, mode='mirror')

    # Remove the padded slices added at the start and end along the Z-axis
    gradient_stack = np.delete(gradient_stack, [0, gradient_stack.shape[2] - 1], axis=2)

    return gradient_stack


# NOT USED - NOT SURE IF USEFUL?
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
        _, BP = bandpass_filter(IN, 2, 30)
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


def positions_batch(params):
    """
    Process a batch of positions from the input tuple. The tuple corresponds to a single frame of .avi file.
    This function will be called on different processors with a different tuple (frame) each time.

    Parameters:
    ----------
    params : tuple
        A tuple containing:
        - hologram (np.ndarray): Frame hologram.
        - median_img (np.ndarray): Median image of frames.
        - ref_index (float): Refractive index.
        - wavelength (float): Wavelength of light.
        - magnification (float): Magnification level (e.g., 40x).
        - start_refocus (float): Starting value for refocus.
        - step_size (float): Step size for propagation.
        - num_steps (int): Number of propagation steps.
        - bandpass_max_px (int): Largest pixel size for bandpass filter.
        - bandpass_min_px (int): Smallest pixel size for bandpass filter.
        - threshold (float): Threshold for gradient stack.
        - peak_min_distance (int): Minimum peak distance.
        - bg_image (np.ndarray): Background image array.
        - use_bg_image (bool): Flag for using background image.

    Returns:
    -------
    list
        A list containing the X, Y, Z positions and intensity arrays (I_FS, I_GS).
    """

    # Unpack params
    (hologram, median_img, ref_index, wavelength, magnification,
     start_refocus, step_size, num_steps, bandpass_max_px, bandpass_min_px,
     threshold, peak_min_distance, bg_image, use_bg_image) = params

    # Calculate scale factor (microns per pixel)
    scale_factor = (magnification / 10) * 0.711  # Adjusted by 0.711 based on system specs

    locs = np.empty((1, 3), dtype=object)
    x_coords, y_coords, z_coords, intensity_fs, intensity_gs = [], [], [], [], []

    # Processing
    propagated_image = rayleigh_sommerfeld_propagator(
        hologram, median_img, ref_index, wavelength, start_refocus,
        scale_factor, step_size, num_steps, True, False, bg_image, use_bg_image,
        bandpass_max_px, bandpass_min_px
    ).astype('float32')

    grad_stack = z_gradient_stack(propagated_image).astype('float32')
    grad_stack[grad_stack < threshold] = 0

    locs[0, 0] = find_3d_positions(grad_stack, peak_min_distance, magnification)
    coords = locs[0, 0].astype('int')

    locs[0, 1] = propagated_image[coords[:, 0], coords[:, 1], coords[:, 2]]
    locs[0, 2] = grad_stack[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Collect results
    x_coords.append(coords[:, 0] * (1 / scale_factor))
    y_coords.append(coords[:, 1] * (1 / scale_factor))
    z_coords.append(coords[:, 2] * step_size)
    intensity_fs.append(locs[0, 1])
    intensity_gs.append(locs[0, 2])

    return [x_coords, y_coords, z_coords, intensity_fs, intensity_gs]


# NOT USED - NOT SURE IF USEFUL?
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
    SRV = TUPLE[5]
    FS = (MPP / 10) * 0.711
    SZ = TUPLE[6]
    NUMSTEPS = TUPLE[7]
    BPL = TUPLE[8]
    BPS = TUPLE[9]
    THRESHOLD = TUPLE[10]
    PMD = TUPLE[11]

    # 0   1    2  3    4    5    6         7
    # zip(IT, MED, n, lam, mpp, sz, numsteps, pmd)

    LOCS = np.empty((1, 3), dtype=object)
    X, Y, Z, I_FS, I_GS = [], [], [], [], []

    GS = modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, True, True)
    GS[GS < THRESHOLD] = 0

    LOCS[0, 0] = find_3d_positions(GS, PMD, MPP, -1)
    A = LOCS[0, 0].astype('int')

    LOCS[0, 1] = GS[A[:, 0], A[:, 1], A[:, 2]]
    LOCS[0, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]

    X.append(LOCS[0, 0][:, 0] * (1 / FS))
    Y.append(LOCS[0, 0][:, 1] * (1 / FS))
    Z.append(LOCS[0, 0][:, 2] * SZ)
    I_FS.append(LOCS[0, 1])
    I_GS.append(LOCS[0, 2])

    return [X, Y, Z, I_FS, I_GS]


def find_3d_positions(gradient_stack, min_peak_distance, magnification, num_particles=-1):
    """
    Find the 3D positions of particles in the gradient stack.

    Parameters:
    ----------
    gradient_stack : np.ndarray
        3D gradient stack.
    min_peak_distance : float
        Minimum distance between peaks in the 2D maximum projection.
    magnification : float
        Magnification factor
    num_particles : int, optional
        Number of particles to detect; -1 to detect all peaks. Default is -1.

    Returns:
    -------
    np.ndarray
        2D array with particle positions (x, y in pixels, z in slice number).
    """

    # Compresses the 3D gradient stack GS into a 2D representation of maximum values along the depth axis
    max_projection = np.max(gradient_stack, axis=-1)

    # Detect peak positions in the 2D projection
    # If num_particles is specified, only that many peaks are kept; if num_particles is set to -1,
    # all peaks above a certain threshold are kept.
    # The result peak_positions is a list of (x, y) coordinates of detected peaks.
    if num_particles == -1:
        peak_positions = peak_local_max(max_projection, min_distance=min_peak_distance)
    else:
        peak_positions = peak_local_max(max_projection, min_distance=min_peak_distance, num_peaks=num_particles)

    # Define neighborhood dimensions based on magnification
    # TODO: should magnification be in MPP? It is currently 40x (i.e. the float '40')
    neighborhood_radius_x = int(magnification / 10)
    neighborhood_radius_y = int(magnification / 10)
    z_intensity_sum = np.empty((gradient_stack.shape[2], len(peak_positions)))

    # Sum intensities around each detected peak in the Z-dimension
    for idx, (peak_x, peak_y) in enumerate(peak_positions):
        sub_region = gradient_stack[
            peak_x - neighborhood_radius_x : peak_x + neighborhood_radius_y,
            peak_y - neighborhood_radius_x : peak_y + neighborhood_radius_y,
            :
        ]
        z_intensity_sum[:, idx] = np.sum(sub_region, axis=(0, 1))

    # Store Z-locations corresponding to maximum intensity for each peak
    z_max_positions_folded = np.empty((len(peak_positions), 1), dtype=object)
    for idx in range(len(peak_positions)):
        max_z_locations = peak_local_max(z_intensity_sum[:, idx], num_peaks=1)
        z_max_positions_folded[idx, 0] = max_z_locations if max_z_locations.size else np.array([[0]])

    # Flatten Z-locations into a simple list
    z_max_positions = [
        [z.item()] if len(z) == 1 else [z.item() for z in z]
        for z in z_max_positions_folded[:, 0]
    ]
    z_max_positions = np.array(z_max_positions)

    # Refine Z-positions using quadratic fitting
    fit_width = 2
    quadratic_fn = lambda a, x: a[0] * x ** 2 + a[1] * x + a[2]
    refined_z_positions = []

    for z_pos in z_max_positions:
        z_index = z_pos[0]
        z_neighborhood = np.arange(z_index - fit_width, z_index + fit_width + 1)
        padded_intensity_sum = np.pad(z_intensity_sum, ((fit_width, fit_width), (0, 0)))
        intensity_values = padded_intensity_sum[z_neighborhood + fit_width, len(refined_z_positions)]

        # Quadratic fitting for sub-pixel Z-position refinement
        coefficients = np.polyfit(z_neighborhood, intensity_values, 2)
        interpolated_z = np.linspace(z_neighborhood[0], z_neighborhood[-1], 10)
        interpolated_values = quadratic_fn(coefficients, interpolated_z)

        max_index = np.argmax(interpolated_values)
        refined_z_positions.append(interpolated_z[max_index])

    # Combine 2D (x, y) positions with refined Z positions into 3D (x, y, z)
    particle_positions_3d = np.insert(np.float16(peak_positions), 2, refined_z_positions, axis=-1)

    return particle_positions_3d  # Returns (x, y in pixels, z in slice number)