from scipy.signal import convolve2d
import numpy as np
import scipy.ndimage.filters as filters
from skimage.color import rgb2gray
from imageio import imread


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


def reduce(im, given_filter):
    """
    A helper function that reduces an image given and a filter
    :param im: An image in matrix form
    :param given_filter: A filter ( An array)
    :return: The reduced image in matrix form
    """
    blur_image = filters.convolve(im.T, given_filter)
    blur_image = filters.convolve(blur_image.T, given_filter)
    blur_image = np.array(blur_image)
    reduced_image_rows = blur_image[::2]
    final_image = reduced_image_rows.T
    final_image = final_image[::2]
    return final_image.T


def filter_calculation(filter_size):
    """
    Calculates the filter
    :param filter_size: An int
    :return:  The calculated filter in a numpy array.
    """
    # calculating the filter
    temp_filter = np.array([1, 1])
    while temp_filter.shape[0] != filter_size:
        temp_filter = np.convolve(temp_filter, np.array([1, 1]))
    final_filter = temp_filter
    # normalize the filter
    final_filter = final_filter / final_filter.sum()
    final_filter = final_filter.reshape(1, final_filter.shape[0])
    return final_filter


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function builds a gaussian pyramid.
    :param im: An image
    :param max_levels: The length of the pyramid
    :param filter_size: The size of the filter
    :return: The pyramid in an array form and the filter.
    """
    if filter_size < 2:
        final_filter = [[1]]
    else:
        final_filter = filter_calculation(filter_size)
    # calculating the pyramid
    pyramid = [np.array(im)]
    for i in range(1, max_levels):
        reduced_image = reduce(pyramid[i-1], final_filter)
        if reduced_image.shape[0] < 16 or reduced_image.shape[1] < 16:
            break
        pyramid.append(reduced_image)
    return pyramid, final_filter


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img
