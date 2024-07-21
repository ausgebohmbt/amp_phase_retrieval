"""
Module to measure the constant amplitude and phase at the SLM.
"""

import os
import glob
import time
import numpy as np
import error_metrics as m, patterns as pt, fitting as ft
import scipy.optimize as opt
import matplotlib.pyplot as plt
from colorama import Fore, Style  # , Back
from experiment import Params


def get_aperture_indices(nx, ny, x_start, x_stop, y_start, y_stop, aperture_width, aperture_height):
    """
    This function calculates a grid of ``nx * ny`` rectangular regions in an array and returns the start and end indices
    of each region. All units are in pixels.

    :param nx: Number of rectangles along x.
    :param ny: Number of rectangles along y.
    :param x_start: Start index for first rectangle along x.
    :param x_stop: End index for last rectangle along x.
    :param y_start: Start index for first rectangle along y.
    :param y_stop: End index for last rectangle along y.
    :param aperture_width: Width of rectangle.
    :param aperture_height: Height of rectangle.
    :return: List with four entries for the start and end index along x and y:
             [idx_start_y, idx_end_y, idx_start_x, idx_end_x]. Each list entry is a vector of length ``nx * ny``
             containing the start/end index for each rectangle along x/y.
    """
    idx_start_x = np.floor(np.linspace(x_start, x_stop - aperture_width, nx)).astype('int')
    idx_end_x = idx_start_x + aperture_width
    idx_start_x = np.tile(idx_start_x, ny)
    idx_end_x = np.tile(idx_end_x, ny)

    idx_start_y = np.floor(np.linspace(y_start, y_stop - aperture_height, ny)).astype('int')
    idx_end_y = idx_start_y + aperture_height
    idx_start_y = np.repeat(idx_start_y, nx)
    idx_end_y = np.repeat(idx_end_y, nx)
    return [idx_start_y, idx_end_y, idx_start_x, idx_end_x]



def recreate_wavefront(pms_obj,
                       aperture_number, aperture_width, img_size,
                       benchmark=False, roi_min_x=16, roi_min_y=16, roi_n=8):
    """
    This function measures the constant phase at the SLM by displaying a sequence of rectangular phase masks on the SLM.
    This scheme was adapted from this Phillip Zupancic's work (https://doi.org/10.1364/OE.24.013881). For details of our
    implementation, read the SI of https://doi.org/10.1038/s41598-023-30296-6.

    :param slm_disp_obj: Instance of your own subclass of ``hardware.SlmBase``.
    :param cam_obj: Instance of your own subclass of ``hardware.CameraBase``.
    :param pms_obj: Instance of your own subclass of ``hardware.Parameters``.
    :param aperture_number: Number of square regions along x/ y.
    :param aperture_width: Width of square regions [px].
    :param img_size: Width of the roi in the camera image [camera pixels].
    :param exp_time: Exposure time.
    :param spot_pos: x/y position of the diffraction spot in th computational Fourier plane [Fourier pixels].
    :param n_avg_frames: Number of camera frames to average per shot.
    :param bool benchmark: Load previously measured constant phase and display it on the SLM to check for flatness.
    :param phi_load_path: Path to previously measured constant phase.
    :param roi_min_x: Aperture column number to display the first phase mask.
    :param roi_min_y: Aperture row number to display the first phase mask.
    :param roi_n: Number of apertures to display (roi_n * roi_n), starting at roi_min_x, roi_min_y.

    :return: Path to measured constant phase at the SLM.
    """

    "prep data Save"
    date_saved = time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())
    path = pms_obj.data_path + date_saved + '_measure_slm_wavefront'
    # os.mkdir(path)

    # current_path = os.getcwd()
    main_path = "E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev"
    path_phase_npy = glob.glob(main_path + "\\amphase_result\\24-07-21_19-56-48_measure_slm_wavefront")

    "set res & stuff"
    cam_pitch = 6.5e-6
    slmX = 1272
    slmY = 1024
    slm_pitch = 12.5e-6
    slm_res = [slmY, slmX]
    # roi_mem = cam_obj.roi
    res_y, res_x = slm_res
    npix = np.min(slm_res)
    border_x = int(np.abs(res_x - res_y) // 2)
    zeros = np.zeros((npix, npix))
    zeros_full = np.zeros((npix, np.max(slm_res)))
    fl = pms_obj.fl


    n_centre = aperture_number ** 2 // 2 + aperture_number // 2 - 1
    n_centre_ref = aperture_number ** 2 // 2 + aperture_number // 2

    idx = range(aperture_number ** 2)
    idx = np.reshape(idx, (aperture_number, aperture_number))
    idx = idx[roi_min_x:roi_min_x + roi_n, roi_min_y:roi_min_y + roi_n]
    idx = idx.flatten()

    slm_idx = get_aperture_indices(aperture_number, aperture_number, border_x, npix + border_x, 0, npix, aperture_width,
                                   aperture_width)

    aperture_width_adj = np.zeros(roi_n ** 2)

    'main lOOp'
    plot_within = False
    aperture_coverage = np.copy(zeros_full)
    iter_num = roi_n ** 2

    saVe_plo = True

    # Fit sine to images
    fit_sine = ft.FitSine(fl, pms_obj.k)

    popt_sv = []
    pcov_sv = []
    perr_sv = []

    print(path_phase_npy)

    img = np.load(path_phase_npy[0] + "\\imgF_iter_{}.npy".format(0))  # ~~
    # x, y = pt.make_grid(img[:, :, 0], scale=cam_pitch)
    x, y = pt.make_grid(img[:, :], scale=cam_pitch)

    print(Fore.LIGHTBLACK_EX + "start em fitz" + Style.RESET_ALL)

    "fit dem d@a"
    for i in range(roi_n ** 2):
        ii = idx[i]
        img_i = np.load(path_phase_npy[0] + "\\imgF_iter_{}.npy".format(i))  # ~~
        # img_i = img[:, :, i]

        dx = (slm_idx[2][ii] - slm_idx[2][n_centre]) * slm_pitch
        dy = (slm_idx[0][ii] - slm_idx[0][n_centre]) * slm_pitch
        fit_sine.set_dx_dy(dx, dy)

        a_guess = np.sqrt(np.max(img_i)) / 2
        p0 = np.array([0, a_guess, a_guess])
        bounds = ([-np.pi, 0, 0], [np.pi, 2 * a_guess, 2 * a_guess])
        x_data = np.vstack((x.ravel(), y.ravel()))
        popt, pcov = opt.curve_fit(fit_sine.fit_sine, x_data, img_i.ravel(), p0, bounds=bounds, maxfev=50000)

        perr = np.sqrt(np.diag(pcov))
        popt_sv.append(popt)
        pcov_sv.append(pcov)
        perr_sv.append(perr)
        print(i + 1)

    dphi = -np.reshape(np.vstack(popt_sv)[:, 0], (roi_n, roi_n))
    dphi_err = np.reshape(np.vstack(perr_sv)[:, 0], (roi_n, roi_n))
    a = np.reshape(np.vstack(popt_sv)[:, 1], (roi_n, roi_n))
    b = np.reshape(np.vstack(popt_sv)[:, 2], (roi_n, roi_n))

    aperture_area = np.reshape(aperture_width_adj, dphi.shape) ** 2
    i_fit = np.abs(a * b)
    i_fit_adj = i_fit / aperture_area
    i_fit_mask = i_fit > 0.01 * np.max(i_fit)

    # Determine phase
    dphi_uw_nopad = pt.unwrap_2d(dphi)
    dphi_uw_notilt = ft.remove_tilt(dphi_uw_nopad)
    pad_roi = ((roi_min_x, aperture_number - roi_n - roi_min_x), (roi_min_y, aperture_number - roi_n - roi_min_y))
    dphi_uw = np.pad(dphi_uw_nopad, pad_roi)

    if benchmark is True:
        rmse = m.rms_phase(dphi_uw_notilt / 2 / np.pi)
        p2v = np.max(dphi_uw_notilt / 2 / np.pi) - np.min(dphi_uw_notilt / 2 / np.pi)
        print('RMS error: lambda /', 1 / rmse)
        print('Peak-to-valley error: lambda /', 1 / p2v)

    dphi_uw_mask = pt.unwrap_2d_mask(dphi, i_fit_mask)
    dphi_uw_mask = np.pad(dphi_uw_mask, pad_roi)

    # # Save data
    # np.save(path + '//dphi', dphi)
    # np.save(path + '//dphi_uw', dphi_uw)
    # np.save(path + '//dphi_err', dphi_err)
    # np.save(path + '//dx', dx)
    # np.save(path + '//dy', dy)
    # np.save(path + '//i_fit', i_fit)
    # np.save(path + '//dphi_uw_mask', dphi_uw_mask)
    # np.save(path + '//i_fit_mask', i_fit_mask)
    # np.save(path + '//t', t)
    # np.save(path + '//popt_sv', popt_sv)
    # np.save(path + '//perr_sv', perr_sv)
    # np.save(path + '//aperture_coverage', aperture_coverage)
    # # np.save(path + '//cal_pos_y', cal_pos_y)

    phErr = plt.figure()
    plt.imshow(dphi_err, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('dphi_err measured phase')
    if saVe_plo:
        plt.show(block=False)
        # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
        phErr.savefig(path +'\\dphi_err.png', dpi=300, bbox_inches='tight',
                    transparent=False)  # True trns worls nice for dispersion thinks I
        plt.pause(0.8)
        plt.close(phErr)
    else:
        plt.show()


    phFig = plt.figure()
    plt.imshow(dphi_uw / np.pi / 2, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Unwrapped measured phase')
    if saVe_plo:
        plt.show(block=False)
        # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
        phFig.savefig(path +'\\Unwrapped.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.8)
        plt.close(phFig)
    else:
        plt.show()

    # img_size = 300
    fit_test = np.reshape(fit_sine.fit_sine(x_data, *popt_sv[-1]), (img_size, img_size))

    fig1 = plt.Figure()
    plt.subplot(121)
    plt.imshow(img[:, :, -1], cmap='turbo')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(122)
    plt.imshow(fit_test, cmap='turbo')
    plt.colorbar(fraction=0.046, pad=0.04)
    if saVe_plo:
        plt.show(block=False)
        # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
        fig1.savefig(path + '\\fit.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(2)
        plt.close(fig1)
    else:
        plt.show()

    # power = np.reshape(aperture_power, (roi_n, roi_n))
    # powah = np.reshape(aperture_powah, (roi_n, roi_n))
    # np.save(path + '//powah', powah)
    # np.save(path + '//power', power)
    # np.save(path + '//imgzz', img)
    #
    # figPow = plt.Figure()
    # plt.subplot(121)
    # plt.imshow(powah, cmap='turbo')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.subplot(122)
    # plt.imshow(power, cmap='turbo')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # if saVe_plo:
    #     plt.show(block=False)
    #     figPow.savefig(path +'\\powaher.png', dpi=300, bbox_inches='tight', transparent=False)
    #     plt.pause(2)
    #     plt.close(figPow)
    # else:
    #     plt.show()

    return path + '//dphi_uw'


pitch = 6.5e-6
pms_obj = Params()
phi_path = recreate_wavefront(pms_obj, 30, 16,
                                         64, roi_min_x=0,
                                         roi_min_y=0, roi_n=30)

# es el finAl
