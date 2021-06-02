import numpy as np
import scipy.io.wavfile
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from imageio import imread
from skimage.color import rgb2gray


def read_image(filename, representation):
    """
    Gets a filename and an int representing which color image we want(greyscale or rgb)
    :param filename: A string
    :param representation: An int. 1 for greyscale and 2 for rgb
    :return: A matrix representing the image
    """
    im = imread(filename)
    im_float = im.astype(np.float64)
    im_float /= 255
    if representation == 1 and len(im.T) == 3:
        return rgb2gray(im_float)

    else:
        return im_float


def DFT(signal):
    """
    Transform fourier on a given array.
    :param signal: An array
    :return: An Array
    """
    signal = np.array(signal)
    N = len(signal)
    if N == 0:
        return np.array([])
    x_values, u_values = np.meshgrid(np.arange(N), np.arange(N))
    matrix_u_w = x_values * u_values
    exp_matrix_u_w = np.exp(-2j*(1/N) * np.pi * matrix_u_w)
    result = np.dot(exp_matrix_u_w, signal)
    return result


def IDFT(fourier_signal):
    """
    This function operates the inverse transform fourier on a given complex array
    :param fourier_signal: A complex array
    :return: An array
    """
    fourier_signal = np.array(fourier_signal)
    N = len(fourier_signal)
    if N == 0:
        return np.array([])
    x_values, u_values = np.meshgrid(np.arange(N), np.arange(N))
    matrix_u_w = x_values * u_values
    exp_matrix_u_w = (1/N) * np.exp(2j * (1 / N) * np.pi * matrix_u_w)
    result = np.dot(exp_matrix_u_w, fourier_signal)
    return result


def DFT2(image):

    """
    Transform fourier on a given  2D array.
    :param image: A 2D array
    :return: A 2D array
    """

    #  Algorithm from class
    image = np.array(image)
    M, N = image.shape
    empty_matrix = np.zeros((M, N), dtype=np.complex128)
    for j in range(N):
        fft_col = DFT(image[:, j])
        empty_matrix[:, j] = fft_col

    result = np.zeros((M, N), dtype=np.complex128)
    for i in range(M):
        fft_row = DFT(empty_matrix[i, :])
        result[i, :] = fft_row
    return result


def IDFT2(fourier_image):
    """
    This function operates the inverse transform fourier on a given complex 2D array
    :param fourier_image: A complex  2D array
    :return: A 2D array
    """
    # Algorithm from class
    fourier_image = np.array(fourier_image)
    M, N = fourier_image.shape
    empty_matrix = np.zeros((M, N), dtype=np.complex128)
    for j in range(N):
        fft_col = IDFT(fourier_image[:, j])
        empty_matrix[:, j] = fft_col

    result = np.zeros((M, N), dtype=np.complex128)
    for i in range(M):
        fft_row = IDFT(empty_matrix[i, :])
        result[i, :] = fft_row
    return result


def change_rate(filename, ratio):
    """
    This function change the sample rate of a given wav file and writes
    it into a new file called change_rate.
    :param filename: A string
    :param ratio: A float
    :return: None
    """
    sample_rate, data = scipy.io.wavfile.read(filename)
    new_sample_rate = sample_rate * ratio
    scipy.io.wavfile.write("change_rate.wav", int(new_sample_rate), data)


def change_samples(filename, ratio):
    """
    This function changes the data from the given wav file by a given algorithm
    :param filename: A string
    :param ratio: A float
    :return: A new data array
    """
    sample_rate, data = scipy.io.wavfile.read(filename)
    new_data = resize(data, ratio)
    new_data = new_data.astype(np.float64)
    scipy.io.wavfile.write("change_samples.wav", sample_rate, new_data)
    return new_data


def resize(data, ratio):
    """
    This function resize a given array by the ratio.
    :param data: An array
    :param ratio: A float
    :return: The new resized array
    """
    dft = DFT(data)
    shifted_dft = np.fft.fftshift(dft)
    result = None
    if ratio == 1:
        result = data
    if ratio < 1:
        number_of_zeroes = int(len(shifted_dft) / ratio) - len(shifted_dft)
        number_of_zeros_to_left = int(number_of_zeroes / 2)
        number_of_zeros_to_right = number_of_zeroes - number_of_zeros_to_left
        zeros_left = np.zeros(number_of_zeros_to_left)
        zeros_right = np.zeros(number_of_zeros_to_right)
        result = np.concatenate([zeros_left, shifted_dft, zeros_right])

    if ratio > 1:
        deleted_elements = len(shifted_dft) - int(len(shifted_dft) / ratio)
        deleted_elements_left = int(deleted_elements/2)
        deleted_elements_right = deleted_elements - deleted_elements_left
        result = shifted_dft[deleted_elements_left: len(shifted_dft) - deleted_elements_right]

    return IDFT(np.fft.ifftshift(result)).astype(data.dtype)


def resize_spectrogram(data, ratio):
    """
    This function changes the speed of a given data by the ratio
    with a spectrogram.
    :param data: An array
    :param ratio: A float
    :return: The new data(An array).
    """
    new_data = stft(data)
    M, N = new_data.shape
    result = [resize(new_data[i, :], ratio) for i in range(M)]
    result = np.array(result)
    result = istft(result)
    return result.astype(data.dtype)


def resize_vocoder(data, ratio):
    """
    This function resize a given a data by a given ratio
    with a helper function.
    :param data: An array
    :param ratio: A float
    :return: The new data(An array).
    """
    spectrogram = stft(data)
    phased_spectrogram = phase_vocoder(spectrogram, ratio)
    return istft(phased_spectrogram)


def conv_der(im):
    """
    This function calculates the derivative of a given image
    with convolution
    :param im: A 2D Array
    :return: A 2D Array
    """
    window = np.array([[0.5, 0, -0.5]])
    x = scipy.signal.convolve2d(im, window, mode="same")
    y = scipy.signal.convolve2d(im, window.T, mode="same")
    magnitude = np.sqrt(np.abs(x) ** 2 + np.abs(y) ** 2)
    return magnitude.astype(im.dtype)


def fourier_der(im):
    """
    This function calculates the derivative of a given image
    with fourier
    :param im: A 2D Array
    :return: A 2D Array
    """
    dft2 = DFT2(im)
    shifted_dft2 = np.fft.fftshift(dft2)
    N, M = im.shape
    u_val = int(N/2)
    v_val = int(M/2)
    u_values = np.arange(-u_val, u_val)[:, None]
    multiplied_u = (u_values * 2j*np.pi)/N
    multiplied_u = multiplied_u * shifted_dft2
    v_values = np.arange(-v_val, v_val)
    multiplied_v = (v_values * 2j*np.pi)/M
    multiplied_v = multiplied_v * shifted_dft2
    dx = IDFT2(np.fft.ifftshift(multiplied_u))
    dy = IDFT2(np.fft.ifftshift(multiplied_v))
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude.astype(np.float64)


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec
