from collections import deque
import itertools

import numpy as np
import scipy
import scipy.optimize as opt


def moving_average(iterable, n=3):
    # moving_average([40, 30, 50, 46, 39, 44]) --> 40.0 42.0 45.0 43.0
    it = iter(iterable)
    d = deque(itertools.islice(it, n - 1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / n


def plot_image(image: np.ndarray, lognorm=False) -> None:
    from matplotlib import pyplot as plt  # type: ignore
    from matplotlib import colors

    norm = colors.LogNorm() if lognorm else None
    plt.imshow(image, norm=norm, cmap='inferno')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()


def argmax(image: np.ndarray) -> tuple[int, ...]:
    return np.unravel_index(image.argmax(), image.shape)


def unimOD(phase):
    mODed = np.mod(phase, 1)
    return mODed


def normalize(im):
    """normalizez 0 2 1"""
    maxi = np.max(im)
    mini = np.min(im)
    norm = ((im - mini) / (maxi - mini))
    return norm


def center_overlay(bg_size_x, bg_size_y, arr):
    """
    ex - center_paste2canVas
    pastes an array to a zero valued background array of the shApe of the slm
    """
    arr = np.asarray(arr)
    aa = np.zeros((bg_size_y, bg_size_x))
    arr_width = arr.shape[1]
    arr_height = arr.shape[0]
    start_x = (bg_size_x - arr_width) // 2
    start_y = (bg_size_y - arr_height) // 2
    aa[start_y:start_y + arr_height, start_x:start_x + arr_width] = arr[:, :]
    return aa


def convertcrazy8(image):
    image = np.asarray(image)
    image = np.round(image, 2)
    conVerted = image.astype(np.uint8)

    return conVerted.tolist()


def center_crop(arr, final_width, final_height):
    """crops an image from its center"""
    width = arr.shape[1]
    height = arr.shape[0]
    startx = (width - final_width) // 2
    starty = (height - final_height) // 2
    arr = arr[starty:starty + final_height, startx:startx + final_width]

    return arr


def tiler(holog, slm_pxl_x, slm_pxl_y):
    holog = np.asarray(holog)
    ratio_imag2slm_x = np.ceil(2 * slm_pxl_x / holog.shape[1])
    ratio_imag2slm_y = np.ceil(2 * slm_pxl_y / holog.shape[0])

    # select larger of dem dimensionz
    dem_ratio = []
    if ratio_imag2slm_x >= ratio_imag2slm_y:
        dem_ratio = int(ratio_imag2slm_x)
    elif ratio_imag2slm_x < ratio_imag2slm_y:
        dem_ratio = int(ratio_imag2slm_y)

    #  tile eeet to an odd dimension in a marvelous(whaaat??) manner
    centeredHoloXtraVaGanZa = []
    if dem_ratio % 2 == 0:
        centeredHoloXtraVaGanZa = np.tile(holog, (dem_ratio + 1, dem_ratio + 1))
    elif dem_ratio % 2 == 1:
        centeredHoloXtraVaGanZa = np.tile(holog, (dem_ratio, dem_ratio))

    return centeredHoloXtraVaGanZa


# Utility functions for array manipulation
def make_grid(im, scale=None):
    """
    Return a xy meshgrid based in an input array, im, ranging from -scal * im.shape[0] // 2 to scal * im.shape[0] // 2.

    :param im: Input array.
    :param scale: Optional scaling factor.
    :return: x and y meshgrid arrays.
    """
    if scale is None:
        scale = 1
    h, w = im.shape
    y_lim, x_lim = h // 2, w // 2

    x, y = np.linspace(-x_lim * scale, x_lim * scale, w), np.linspace(-y_lim * scale, y_lim * scale, h)
    x, y = np.meshgrid(x, y)
    return x, y


def gaussian_int(x, y, dx, dy, sig_x, sig_y=None, a=1, c=0):
    """
    2D Gaussian.

    :param x: X meshgrid.
    :param y: Y meshgrid.
    :param dx: X-offset of Gaussian.
    :param dy: Y-offset of Gaussian.
    :param sig_x: X width of Gaussian.
    :param sig_y: Y width of Gaussian
    :param a: Amplitude.
    :param c: Offset.
    :return: 2D Gaussian.
    """
    if sig_y is None:
        sig_y = sig_x

    return a * np.exp(-0.5 * ((x - dx) ** 2 / sig_x ** 2 + (y - dy) ** 2 / sig_y ** 2)) + c


def gaussian(xy, *args):
    """
    Gaussian fit function.

    :param xy: x, y coordinate vectors.
    :param args: Fitting parameters passed to patterns.gaussian.
    :return: Gaussian.
    """
    x, y = xy
    arr = gaussian_int(x, y, *args)
    return arr


def fit_gaussian(img, dx=None, dy=None, sig_x=15, sig_y=15, a=None, c=0, blur_width=10, xy=None):
    """
    Fits a 2D Gaussian to an image. The image s blurred using a Gaussian filer before fitting.

    :param img: Input image.
    :param dx: X-offset of Gaussian [px].
    :param dy: Y-offset of Gaussian [px].
    :param sig_x: X-width of Gaussian [px].
    :param sig_y: -width of Gaussian [px].
    :param a: Amplitude.
    :param c: Offset.
    :param blur_width: Width of Gaussian blurring kernel [px].
    :param xy: X, Y meshgrid. If not specified, pixel coordinates are used.
    :return: Fitting parameters, parameter errors.
    """
    if xy is None:
        x, y = make_grid(img)
    else:
        x, y = xy
    x_data = np.vstack((x.ravel(), y.ravel()))
    img_blur = scipy.ndimage.gaussian_filter(img, blur_width)
    if dx is None or dy is None:
        dy, dx = np.unravel_index(np.argmax(img_blur), img.shape)
        dx -= img.shape[1] / 2
        dy -= img.shape[0] / 2
    if a is None:
        a = np.max(img_blur)
    # Define initial parameter guess.
    p0 = [(dx, dy, sig_x, sig_y, a, c)]
    popt, pcov = opt.curve_fit(gaussian, x_data, img.ravel(), p0, maxfev=10000)

    # Calculate errors
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def closest(lst, K) -> tuple:
    """searches list lst for the closest value to K.

     return:
         tuple with index and closest value existing the list"""
    return min(enumerate(lst), key=lambda x: abs(x[1]-K))


def closest_arr(arr, K) -> tuple:
    diff = np.absolute(arr - K)
    index = diff.argmin()
    return index, arr[index]


def draw_circle(arraysz, radious):
    xx, yy = np.mgrid[-radious:radious + 1, -radious:radious + 1]
    circle = xx ** 2 + yy ** 2 <= radious ** 2
    # bg = np.zeros((arraysz, arraysz))
    arrr = center_overlay(arraysz, arraysz, circle.astype(int))
    return arrr


def do_overlay(bg_size_x, bg_size_y, arr, x0, y0):
    """
    ex - center_paste2canVas
    pastes an array to a zero valued background array of the shApe of the slm
    """
    arr = np.asarray(arr)
    aa = np.zeros((bg_size_y, bg_size_x))
    arr_width = arr.shape[1]
    arr_height = arr.shape[0]
    start_x = ((bg_size_x - arr_width) // 2) + x0
    start_y = ((bg_size_y - arr_height) // 2) + y0
    aa[start_y:start_y + arr_height, start_x:start_x + arr_width] = arr[:, :]
    return aa


def draw_circle_displaced(arraysz, radious, x0, y0):
    """creates a circle of given radious and pastes it
        on a canvas of size arraysz displaced by x0, y0"""
    xx, yy = np.mgrid[-radious:radious + 1, -radious:radious + 1]
    circle = xx ** 2 + yy ** 2 <= radious ** 2
    arrr = do_overlay(arraysz, arraysz, circle.astype(int), x0, y0)
    return arrr


def draw_n_paste_circle(canvas, radious, x0, y0, value):
    xx, yy = np.mgrid[-radious:radious + 1, -radious:radious + 1]
    circle = xx ** 2 + yy ** 2 <= radious ** 2
    circle = circle * value
    # arrr = do_overlay(arraysz, arraysz, circle.astype(int), x0, y0)
    arr_width = circle.shape[1]
    arr_height = circle.shape[0]
    start_x = ((canvas.shape[1] - arr_width) // 2) + x0
    start_y = ((canvas.shape[0] - arr_height) // 2) + y0
    canvas[start_y:start_y + arr_height, start_x:start_x + arr_width] = circle[:, :]
    return canvas


def separate_graph_regions(stack, graph):
    """separate_graph_regions_of_dOOm , given a stack it is separated into three
        different regions.
        Gaussian fits are used to find the center of the peaks on a second input graph
        which is the first layer of the stack where the background has been removed"""

    approx_nulla = 1568
    approx_primus = 1010
    approx_secundus = 446
    height_o_graph_o = graph.shape[0]
    half_height_o_graph_o = int(graph.shape[0] // 2)
    extend = 42
    half_height_o_graph_o_extended = half_height_o_graph_o + extend
    from_primus = approx_primus - half_height_o_graph_o_extended
    from_nulla = approx_nulla - half_height_o_graph_o_extended
    from_secundus = approx_secundus - half_height_o_graph_o_extended

    print("graph height: {}".format(half_height_o_graph_o))
    graph_primus = stack[:height_o_graph_o, from_primus:approx_primus + half_height_o_graph_o_extended]
    graph_nulla = stack[:height_o_graph_o, from_nulla:approx_nulla + half_height_o_graph_o_extended]
    graph_secundus = stack[:height_o_graph_o, from_secundus:approx_secundus + half_height_o_graph_o_extended]
    print("shape graph_secundus: {}".format(graph_secundus.shape))
    print("shape graph_primus ref: {}".format(graph_primus.shape))
    print("shape graph_nulla ref: {}".format(graph_nulla.shape))

    " define centers of radii and roiz "
    # regione primae
    yshape_primus, xshape_primus = graph_primus.shape
    print("fit input x, y: {}, {}".format(xshape_primus, yshape_primus))
    p_opt_msr, p_err_msr = fit_gaussian(graph_primus)
    popt_clb_msr = p_opt_msr[:2]
    print("popt_clb_msr, p_opt_msr: {}, {}".format(popt_clb_msr, p_opt_msr))
    x_0_primus = int(popt_clb_msr[0] + xshape_primus // 2)  # x_0 = int(popt_clb[0] + nx // 2)
    y_0_primus = int(popt_clb_msr[1] + yshape_primus // 2)  # y_0 = int(popt_clb[1] + ny // 2)
    print('2d gau fit result of 1st order region: x0 = {}, y0 = {}'.format(x_0_primus, y_0_primus))

    # regione nulla
    yshape_nulla, xshape_nulla = graph_nulla.shape
    print("fit input x, y: {}, {}".format(yshape_nulla, xshape_nulla))
    p_opt_nulla, p_err_nulla = fit_gaussian(graph_nulla)
    popt_clb_nulla = p_opt_nulla[:2]
    print("popt_clb_msr, p_opt_msr: {}, {}".format(popt_clb_nulla, p_opt_nulla))
    x_0_nulla = int(popt_clb_nulla[0] + xshape_nulla // 2)  # x_0 = int(popt_clb[0] + nx // 2)
    y_0_nulla = int(popt_clb_nulla[1] + yshape_nulla // 2)  # y_0 = int(popt_clb[1] + ny // 2)
    print('2d gau fit result of 0th order region: x0 = {}, y0 = {}'.format(x_0_nulla, y_0_nulla))

    # secunda regione
    yshape_secundus, xshape_secundus = graph_secundus.shape
    print("fit input x, y: {}, {}".format(yshape_secundus, xshape_secundus))
    p_opt_secundus, p_err_secundus = fit_gaussian(graph_secundus)
    popt_clb_secundus = p_opt_secundus[:2]
    print("popt_clb_msr, p_opt_msr: {}, {}".format(popt_clb_secundus, p_opt_secundus))
    x_0_secundus = int(popt_clb_secundus[0] + xshape_secundus // 2)  # x_0 = int(popt_clb[0] + nx // 2)
    y_0_secundus = int(popt_clb_secundus[1] + yshape_secundus // 2)  # y_0 = int(popt_clb[1] + ny // 2)
    print('2d gau fit result of 2nd order region: x0 = {}, y0 = {}'.format(x_0_secundus, y_0_secundus))

    print('primus')
    there_primus = from_primus + x_0_primus
    from_new_primus = there_primus - half_height_o_graph_o_extended
    coord_diff = from_new_primus - from_primus
    stack_primus_crop = stack[:, from_new_primus:
                                 (there_primus + half_height_o_graph_o_extended) + 1]

    print('nulla')
    there_nulla = from_nulla + x_0_nulla
    from_new_nulla = there_nulla - half_height_o_graph_o_extended
    coord_diff_nulla = from_new_nulla - from_nulla
    stack_nulla_crop = stack[:, from_new_nulla:
                                 (there_nulla + half_height_o_graph_o_extended) + 1]

    print('secundus')
    there_secundus = from_secundus + x_0_secundus
    from_new_secundus = there_secundus - half_height_o_graph_o_extended
    coord_diff_secundus = from_new_secundus - from_secundus
    stack_secundus_crop = stack[:, from_new_secundus:
                                 (there_secundus + half_height_o_graph_o_extended) + 1]

    import matplotlib.pyplot as plt
    plt.subplot(231)
    plt.vlines(x_0_primus, colors='m', ymin=0, ymax=stack.shape[0], linewidth=1.4)
    plt.imshow(graph_primus, cmap='inferno')
    plt.title("from_primus {}, there {}".format(from_primus, there_primus))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(232)
    plt.vlines(x_0_primus - coord_diff, colors='g', ymin=0, ymax=stack.shape[0], linewidth=1.4)
    plt.imshow(stack_primus_crop, cmap='inferno')
    plt.title("new point {}".format(x_0_primus + coord_diff))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(233)
    plt.vlines(x_0_nulla, colors='m', ymin=0, ymax=stack.shape[0], linewidth=1.4)
    plt.imshow(graph_nulla, cmap='inferno')
    plt.title("from_nulla {}, there_nulla {}".format(from_nulla, there_nulla))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(234)
    plt.vlines(x_0_nulla - coord_diff_nulla, colors='g', ymin=0, ymax=stack.shape[0], linewidth=1.4)
    plt.imshow(stack_nulla_crop, cmap='inferno')
    plt.title("new point nulla {}".format(x_0_nulla + coord_diff_nulla))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(235)
    plt.vlines(x_0_secundus, colors='m', ymin=0, ymax=stack.shape[0], linewidth=1.4)
    plt.imshow(graph_secundus, cmap='inferno')
    plt.title("from_secundus {}, there_secundus {}".format(from_secundus, there_secundus))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(236)
    plt.vlines(x_0_secundus - coord_diff_secundus, colors='g', ymin=0, ymax=stack.shape[0], linewidth=1.4)
    plt.imshow(stack_secundus_crop, cmap='inferno')
    plt.title("new point secundus {}".format(x_0_secundus + coord_diff_secundus))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

    return stack_primus_crop #, graph_rf_crop




# es el final
