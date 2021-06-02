import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from skimage.color import rgb2gray
from imageio import imread
import os


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


def expand(im, given_filter):
    """
    This function expands an image with a given filter by an
    algorithm given in class
    :param im: An image in matrix form
    :param given_filter: An array
    :return: The expanded image
    """
    im = np.array(im)
    expanded_image = np.zeros((im.shape[0], 2 * im.shape[1]))
    expanded_image[:, ::2] = im
    final_expanded_image = np.zeros((2*expanded_image.shape[0], expanded_image.shape[1]))
    final_expanded_image[::2] = expanded_image
    final_expanded_image = filters.convolve(final_expanded_image.T, given_filter)
    return filters.convolve(final_expanded_image.T, given_filter)


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


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    This function builds a laplacian pyramid by using the function above and the
    algorithm taught in class.
    :param im: An image
    :param max_levels: The length of the pyramid
    :param filter_size: The size of the filter
    :return: The pyramid in an array form and the filter.
    """
    gaussian_pyramid, final_filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplace_pyramid = []
    current_gauss = gaussian_pyramid[0]
    for i in range(1,  max_levels):
        if i > len(gaussian_pyramid) - 1:
            break
        expanded_image = expand(gaussian_pyramid[i], 2*final_filter)
        laplace_pyramid.append(current_gauss - expanded_image)
        current_gauss = gaussian_pyramid[i]
    laplace_pyramid.append(gaussian_pyramid[-1])
    return laplace_pyramid, final_filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Takes a laplacian pyramid and turns it into an image
    :param lpyr: A laplacian pyramid ( A list)
    :param filter_vec: A filter vector An array
    :param coeff: A list of coefficients
    :return: The image
    """
    temp_floor = lpyr[-1]
    temp_floor = np.array(temp_floor)
    temp_floor = coeff[-1] * temp_floor
    result = temp_floor
    for i in range(len(lpyr) - 2, -1, -1):
        result = expand(result, 2 * filter_vec)
        temp = np.array(lpyr[i]) * coeff[i]
        result += temp
    return result


def clip_image(image):
    """
    This function stretches an image values
    :param image: A matrix
    :return: The stretched image
    """
    minimum_value = np.min(image)
    maximum_value = np.max(image)
    return (image - minimum_value)/(maximum_value - minimum_value)


def render_pyramid(pyr, levels):
    """
    This function renders a pyramid into an image
    :param pyr: A list
    :param levels: The amount of levels in the pyramid
    :return: The image with all the levels
    """
    final_image = [clip_image(np.array(pyr[0]))]
    for i in range(1, levels):
        np_pyr = np.array(pyr[i])
        stretched_level = clip_image(np_pyr)
        constant = pyr[0].shape[0] - pyr[i].shape[0]
        with_padding = np.pad(stretched_level, [(0, constant), (0, 0)], "constant")
        final_image.append(with_padding)
    images_combined = np.hstack(final_image)
    return images_combined


def display_pyramid(pyr, levels):
    """
    This function displays the pyramid
    :param pyr: A list of images
    :param levels: The length of the list
    :return: Nothing
    """
    rendered_pyramid = render_pyramid(pyr, levels)
    plt.imshow(rendered_pyramid, cmap="gray")
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Blends two pyramids together
    :param im1: A matrix
    :param im2: A matrix
    :param mask: A boolean image
    :param max_levels: The max levels of pyramid (An int)
    :param filter_size_im: The image filter size
    :param filter_size_mask: The mask filter size
    :return: The blended image
    """
    mask = mask.astype(np.float64)
    laplacian_pyr_im1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    laplacian_pyr_im2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    gaussian_pyr = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[0]
    final_laplacian = []
    for i in range(len(gaussian_pyr)):
        final_laplacian.append(gaussian_pyr[i] * laplacian_pyr_im1[i] + (1-gaussian_pyr[i]) * laplacian_pyr_im2[i])
    image_blended = laplacian_to_image(final_laplacian, filter_vec, np.ones(max_levels))
    np.clip(image_blended, 0, 1, out=image_blended)
    return image_blended


def relpath(filename):
    """
    :param filename: A string
    :return: The real path of a file name.
    """
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    """
    A blending example
    :return: All the different images
    """
    im1 = read_image(relpath("externals/astros.jpg"), 2)
    im2 = read_image(relpath("externals/ufo.jpg"), 2)
    mask = read_image(relpath("externals/astros_mask.jpg"), 1)
    mask = np.round(mask)
    mask = mask.astype(np.bool)
    blended_image = []
    for i in range(3):
        blended_image.append(pyramid_blending(im1[:, :, i], im2[:, :, i], mask, 2, 6, 6))
    final_blended_image = np.dstack(blended_image)
    plot_images(im1, im2, mask, final_blended_image)
    return im1, im2, mask, final_blended_image


def blending_example2():
    """
    A blending example
    :return: All the different images
    """
    im1 = read_image(relpath("externals/real_giraffe.jpg"), 2)
    im2 = read_image(relpath("externals/whitehouse.jpg"), 2)
    mask = read_image(relpath("externals/giraffe_mask.jpg"), 1)
    blended_image = []
    mask = np.round(mask)
    mask = mask.astype(np.bool)
    for i in range(3):
        blended_image.append(pyramid_blending(im1[:, :, i], im2[:, :, i], mask, 2, 6, 6))
    final_blended_image = np.dstack(blended_image)
    plot_images(im1, im2, mask, final_blended_image)
    return im1, im2, mask, final_blended_image


def plot_images(im1, im2, mask, blended_image):
    """
    Plots all the images in the way wanted
    :param im1: An image
    :param im2: An image
    :param mask: An image
    :param blended_image: The final blended image
    :return: nothing
    """
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(2, 2, 1)
    plt.imshow(im1, cmap=plt.cm.gray)
    fig.add_subplot(2, 2, 2)
    plt.imshow(im2, cmap=plt.cm.gray)
    fig.add_subplot(2, 2, 3)
    plt.imshow(mask, cmap=plt.cm.gray)
    fig.add_subplot(2, 2, 4)
    plt.imshow(blended_image, cmap=plt.cm.gray)
    plt.show()
