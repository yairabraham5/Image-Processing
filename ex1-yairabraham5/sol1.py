import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import skimage.color
from imageio import imread

# the matrix to transfer from rgb to yiq and vice versa
CONSTANT_MATRIX = np.array([[0.299, 0.587, 0.114],
                            [0.596, -0.275, -0.321],
                            [0.212, -0.523, 0.311]])


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


def imdisplay(filename, representation):
    """
    Gets an image and shows it on the screen
    :param filename: A string ( name of the image)
    :param representation: An int. 1 for greyscale and 2 for rgb
    :return: None
    """
    image = read_image(filename, representation)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()


def rgb2yiq(imRGB):
    """
    changing the color from rgb to yiq using
    the constant matrix I was given
    :param imRGB: A matrix that is shaped (x*y*3)
    :return: a yiq matrix
    """
    return np.dot(imRGB, CONSTANT_MATRIX.T)


def yiq2rgb(imYIQ):
    """
    changing the color from yiq to rgb using
    the constant matrix I was given
    :param imYIQ: A matrix that is shaped (x*y*3)
    :return: a rgb matrix
    """

    return np.dot(imYIQ, (np.linalg.inv(CONSTANT_MATRIX)).T)


def histogram_image(image):
    """
    This is the helper function for histogram equalizer that differentiates
    greyscale image from rgb
    :param image: A matrix representing the image
    :return: if the image is rgb then returns the histogram for y and True
    else returns the histogram for the picture and False
    """
    is_rgb = False
    if len(image.T) == 3:  # check to see if the image is rgb
        image = rgb2yiq(image)
        is_rgb = True

    original_hist = None
    if is_rgb:
        y = image[:, :, 0]
        y *= 255
        y = np.round(y)
        original_hist, axis = np.histogram(y, 256, (0, 255))

    if not is_rgb:
        image = (image*255).round().astype(np.uint8)
        original = image
        original_hist, axis = np.histogram(original, 256, (0, 255))

    return original_hist, is_rgb


def histogram_equalize(im_orig):
    """
    Gets a Matrix of an image and equalizes the histogram
    :param im_orig: A matrix representing the image
    :return: The new equalized image, the original histogram, the new histogram
    """
    original = im_orig.copy()
    original_hist, is_rgb = histogram_image(original)
    # if is_rgb:
    original = (original*255).round().astype(np.uint8)
    cumulative_hist = np.cumsum(original_hist)
    # find m (first grey scale that is non zero)
    non_zeroes = np.nonzero(cumulative_hist)
    m = non_zeroes[0][0]

    T = 255 * ((cumulative_hist - cumulative_hist[m]) / (cumulative_hist[-1] - cumulative_hist[m]))
    T = np.round(T)
    image = T[original.astype(np.uint8)]
    new_hist, bins = np.histogram(image, 256, (0, 255))
    image_float = image.astype(np.float64)
    image_float /= 255
    if is_rgb:
        original = rgb2yiq(im_orig)
        original[:, :, 0] = image_float[:, :, 0]
        return yiq2rgb(original), original_hist, new_hist

    else:
        return image_float, original_hist, new_hist


def quantize(im_orig, n_qaunt, n_iter):
    """
    This function quantize a given image
    :param im_orig: The image in a matrix form
    :param n_qaunt: The number of colors we are given
    :param n_iter: The number of iterations we want
    :return: A tuple. The new image in a matrix form and the error rate of
    each iteration in an array.
    """
    original = im_orig.copy()
    original_histogram, is_rgb = histogram_image(original)
    if is_rgb:
        original = np.floor(original * 255)
    error_list = []
    q_list = [0] * n_qaunt
    cumulative_hist = np.cumsum(original_histogram)

    z_list = [0] + [np.where(cumulative_hist >= i * (cumulative_hist[-1] / n_qaunt))[0][0]
                    for i in range(1, n_qaunt)] + [255]

    for j in range(n_qaunt):
        lower_val = z_list[j] + 1
        upper_val = z_list[j + 1]
        g_val = np.arange(lower_val, upper_val + 1)
        q_list[j] = np.sum(g_val * original_histogram[lower_val: upper_val + 1]) / \
                    np.sum(original_histogram[lower_val: upper_val + 1])

    for i in range(n_iter):

        # compute the new z values
        temp_z = [0] + [(q_list[i-1] + q_list[i])/2 for i in range(1, n_qaunt)] + [255]

        # check z values
        if temp_z == z_list:
            break
        elif temp_z != z_list:
            z_list = temp_z
        # compute the new q values

        for j in range(n_qaunt):
            lower_val = int(z_list[j]) + 1
            upper_val = int(z_list[j + 1])
            g_val = np.arange(lower_val, upper_val + 1)
            q_list[j] = np.sum(g_val * original_histogram[lower_val: upper_val + 1]) /\
                        np.sum(original_histogram[lower_val: upper_val + 1])

        # compute the error value of this iteration
        error_rate = 0
        for k in range(n_qaunt):
            current_q = q_list[k]
            lower_val = int(z_list[k]) + 1
            upper_val = int(z_list[k + 1])
            g_values = np.arange(lower_val, upper_val+1)
            error_rate += np.sum(np.power(current_q-g_values, 2)*original_histogram[lower_val: upper_val + 1])

        error_list.append(error_rate)

    new_hist = np.array([0] * 256)

    for m in range(n_qaunt):
        lower_val = int(z_list[m])
        upper_val = int(z_list[m+1])
        new_hist[lower_val: upper_val+1] = np.floor(q_list[m])

    image = new_hist[original.astype(np.uint64)]

    image_float = image.astype(np.float64)
    image_float /= 255
    if is_rgb:
        im_orig = rgb2yiq(im_orig)
        im_orig[:, :, 0] = image_float[:, :, 0]
        return yiq2rgb(im_orig), error_list
    else:
        return image_float, error_list


images = []
jer_bw = read_image(r"externals/jerusalem.jpg", 1)
images.append((jer_bw, "jerusalem grayscale"))
jer_rgb = read_image(r"externals/jerusalem.jpg", 2)
images.append((jer_rgb, "jerusalem RGB"))
low_bw = read_image(r"externals/low_contrast.jpg", 1)
images.append((low_bw, "low_contrast grayscale"))
low_rgb = read_image(r"externals/low_contrast.jpg", 2)
images.append((low_rgb, "low_contrast RGB"))
monkey_bw = read_image(r"externals/monkey.jpg", 1)
images.append((monkey_bw, "monkey grayscale"))
monkey_rgb = read_image(r"externals/monkey.jpg", 2)
images.append((monkey_rgb, "monkey RGB"))


def test_rgb2yiq_and_yiq2rgb(im, name):
    """
    Tests the rgb2yiq and yiq2rgb functions by comparing them to the built in ones in the skimage library.
    Allows error to magnitude of 1.e-3 (Difference from built in functions can't be bigger than 0.001).
    :param im: The image to test on.
    :param name: Name of image.
    :return: 1 on success, 0 on failure.
    """
    imp = rgb2yiq(im)
    off = skimage.color.rgb2yiq(im)

    if not np.allclose(imp, off, atol=1.e-3):
        print("ERROR: in rgb2yiq on image '%s'" % name)
        return 0
    imp2 = yiq2rgb(imp)
    off2 = skimage.color.yiq2rgb(off)
    if not np.allclose(imp2, off2, atol=1.e-3):
        print("ERROR: in yiq2rgb on image '%s'" % name)
        return 0
    print("passed conversion test on '%s'" % name)
    return 1


for im in images:
    if len(im[0].shape) == 3:
        result = test_rgb2yiq_and_yiq2rgb(im[0], im[1])
        if not result:
            print("=== Failed Conversion Test ===")
            break


def display_all(im, add_bonus):
    if len(im.shape) == 3 and add_bonus:
        fig, a = plt.subplots(nrows=3, ncols=2)
    else:
        fig, a = plt.subplots(nrows=2, ncols=2)

    # adds the regular image
    a[0][0].imshow(im, cmap=plt.cm.gray)
    a[0][0].set_title(r"original image")

    # adds the quantified image
    quant = quantize(im, 3, 10)[0]
    a[0][1].imshow(quant, cmap=plt.cm.gray)
    a[0][1].set_title(r"quantize to 3 levels, 10 iterations")

    # adds the histogram equalized image
    hist = histogram_equalize(im)[0]
    a[1][0].imshow(hist, cmap=plt.cm.gray)
    a[1][0].set_title("histogram equalization")

    # adds quantization on histogram equalized image
    hist_quant = quantize(hist, 6, 10)[0]
    a[1][1].imshow(hist_quant, cmap=plt.cm.gray)
    a[1][1].set_title("quantize on equalization")

    # adds the bonus image
    # if len(im.shape) == 3 and add_bonus:
    #     a[2][0].imshow(quantize_rgb(im, 3))
    #     a[2][0].set_title(r"bonus quantize_rgb")

    plt.show()









if __name__ == '__main__':
    for im in images:
        # change "False" to "True" if you wish to add the bonus task to the print
        display_all(im[0], False)
    # im = read_image("monkey.jpg", 2)
    # yiq = rgb2yiq(im)
    # img = yiq2rgb(yiq)
    # x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
    # x_normalize = x.astype(np.float64)
    # x_normalize /= 255
    # grad = np.tile(x_normalize, (256, 1))
    # result = histogram_equalize(grad)
    #
    # plt.imshow(result[0], cmap=plt.cm.gray)
    # plt.show()
    # after_quant = quantize(result[0], 5, 5)
    # # plt.show()
    # plt.imshow(after_quant[0], cmap=plt.cm.gray)
    # plt.show()
    # print(after_quant[1])
    # hist = histogram_image(after_quant[0])
    # print(hist[0])

    # imdisplay("monkey.jpg", 2)
    # imdisplay("monkey.jpg", 1)
