"""
Module to measure the constant amplitude and phase at the SLM.
"""

import os
import time

import numpy
import numpy as np
import hardware as hw
import error_metrics as m, patterns as pt, fitting as ft
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from peripheral_instruments.thorlabs_shutter import shutter as sh
from colorama import Fore, Style  # , Back
from slm.phase_generator import phagen as phuzGen
import copy


def normalize(im):
    """normalizez 0 2 1"""
    maxi = np.max(im)
    mini = np.min(im)
    norm = ((im - mini) / (maxi - mini))
    return norm


def find_camera_position(slm_disp_obj, cam_obj, pms_obj, lin_phase, exp_time=100, aperture_diameter=25, roi=[500, 500]):
    """
    This function generates a spot on the camera by displaying a circular aperture on the SLM containing a linear phase
    gradient. The position of the spot is found by fitting a Gaussian to the camera image.

    :param slm_disp_obj: Instance of your own subclass of ``hardware.SlmBase``
    :param cam_obj:
    :param pms_obj:
    :param npix: Number of used SLM pixels
    :param lin_phase: x and y gradient of the linear phase
    :param cam_name: Name of the camera to be used
    :param exp_time: Exposure time
    :param aperture_diameter: Diameter of the circular aperture
    :param roi: Width and height of the region of interest on the camera to remove the zeroth-order diffraction spot
    :return: x and y coordinates of the spot on the camera
    """
    resolution_y, resolution_x = slm_disp_obj.res
    zeros = np.zeros((resolution_y, resolution_y))
    slm_phase = pt.init_phase(zeros, slm_disp_obj, pms_obj, lin_phase=lin_phase)
    circ_aperture = pt.circ_mask(zeros, 0, 0, aperture_diameter / 2)

    #
    # fig = plt.figure()
    # # plt.imshow(slm_phase * circ_aperture, cmap='inferno')
    # plt.imshow(slm_phase, cmap='inferno')
    # plt.colorbar()
    # plt.title('slm_phase * circ_aperture')
    # plt.show()
    #
    # fig = plt.figure()
    # # plt.imshow(slm_phase * circ_aperture, cmap='inferno')
    # plt.imshow(np.flipud(np.fliplr(slm_phase)), cmap='inferno')
    # plt.colorbar()
    # plt.title('slm_phase * circ_aperture')
    # plt.show()
    #
    # fig = plt.figure()
    # # plt.imshow(slm_phase * circ_aperture, cmap='inferno')
    # plt.imshow(np.fliplr(slm_phase), cmap='inferno')
    # plt.colorbar()
    # plt.title('slm_phase TRANS')
    # plt.show()

    slm_phase = np.flipud(np.fliplr(slm_phase))

    # Display phase pattern on SLM
    slm_disp_obj.display(slm_phase)
    # slm_disp_obj.display(slm_phase * circ_aperture)

    # Take camera image
    # cam_obj.start()
    # img = cam_obj.take_image()
    # img = cam_obj.get_image(exp_time)
    # cam_obj.stop()
    cam_obj.take_image()
    img = cam_obj.last_frame

    fig = plt.figure()
    plt.imshow(img, cmap='inferno', vmax=1000)
    plt.colorbar()
    plt.title('slm_phase * circ_aperture IMG')
    plt.show()

    cam_roi_pos = [650, 850]
    cam_roi_sz = [400, 400]
    cam_obj.roi_set_roi(int(cam_roi_pos[0] * cam_obj.bin_sz), int(cam_roi_pos[1] * cam_obj.bin_sz),
                        int(cam_roi_sz[0] * cam_obj.bin_sz), int(cam_roi_sz[1] * cam_obj.bin_sz))

    cam_obj.take_image()
    img = cam_obj.last_frame

    fig = plt.figure()
    plt.imshow(img, cmap='inferno', vmax=1000)
    plt.colorbar()
    plt.title('ROI IMG')
    plt.show()


    # Mask to crop camera image (removes the zeroth-order diffraction spot)
    crop_mask = pt.rect_mask(img, 0, 0, roi[0], roi[1])

    # fig = plt.figure()
    # plt.imshow(crop_mask, cmap='inferno')
    # plt.title('crop_mask')
    # plt.colorbar()
    # plt.show()
    #
    # fig = plt.figure()
    # plt.imshow(img * crop_mask, cmap='inferno', vmax=1000)
    # plt.title('img * crop_mask')
    # plt.colorbar()
    # plt.show()

    # Fit Gaussian to camera image
    p_opt, p_err = ft.fit_gaussian(img * crop_mask)
    return p_opt[:2], img


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


def measure_slm_intensity(slm_disp_obj, cam_obj, pms_obj, aperture_number, aperture_width, exp_time, spot_pos, roi_width):
    """
    This function measures the intensity profile of the laser beam incident onto the SLM by displaying a sequence of
    rectangular phase masks on the SLM. The phase mask contains a linear phase which creates a diffraction spot on the
    camera. The position of the phase mask is varied across the entire area of the SLM and the intensity of each
    diffraction spot is measured using the camera. Read the SI of https://doi.org/10.1038/s41598-023-30296-6 for
    details.

    :param slm_disp_obj: Instance of your own subclass of ``hardware.SlmBase``.
    :param cam_obj: Instance of your own subclass of ``hardware.CameraBase``.
    :param aperture_number: Number of square regions along x/ y.
    :param aperture_width: Width of square regions [px].
    :param exp_time: Exposure time.
    :param spot_pos: x/y position of the diffraction spot in th computational Fourier plane [Fourier pixels].
    :param roi_width: Width of the region of interest on the camera [camera pixels].
    :return:
    """
    roi_mem = cam_obj.roi_is
    date_saved = time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())
    path = pms_obj.data_path + date_saved + '_' + 'measure_slm_intensity'
    os.mkdir(path)

    res_y, res_x = slm_disp_obj.res
    border_x = int(np.abs(res_x - res_y) // 2)
    npix = np.min(slm_disp_obj.res)

    zeros = np.zeros((npix, npix))
    zeros_full = np.zeros((npix, np.max(slm_disp_obj.res)))

    lin_phase = np.array([-spot_pos, -spot_pos])
    slm_phase4me = pt.init_phase(np.zeros((1024, 1272)), slm_disp_obj,
                         pms_obj, lin_phase=lin_phase)
    slm_phaseOUT = pt.init_phase(np.zeros((aperture_width, aperture_width)), slm_disp_obj,
                         pms_obj, lin_phase=lin_phase)

    # slm_phase = pt.init_phase(np.zeros((aperture_width, aperture_width)), slm_disp_obj, pms_obj, lin_phase=lin_phase)
    slm_phase = np.remainder(slm_phaseOUT, 2 * np.pi)
    slm_phase4me = np.remainder(slm_phase4me, 2 * np.pi)
    slm_phase4me = np.fliplr(slm_phase4me)
    # slm_phase = np.remainder(slm_phase, 2 * np.pi)
    slm_phase = slm_phase4me
    # slm_phase = np.flipud(np.fliplr(slm_phase))
    slm_idx = get_aperture_indices(aperture_number, aperture_number, border_x, npix + border_x, 0, npix, aperture_width,
                                   aperture_width)

    # Display central sub-aperture on SLM and check if camera is over-exposed.
    # i = (aperture_number ** 2) // 2 - aperture_number // 2
    # phi_centre = np.zeros_like(zeros)
    # phi_centre = slm_phase
    # slm_phaseNOR = normalize(slm_phase)

    print("mk phuz 4 skm")
    phuzGen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": True}
    phuzGen.linear_grating()
    # phuzGen.grat = slm_phaseNOR
    phuzGen._make_full_slm_array()
    phi_centre = phuzGen.final_phuz
    # phi_centre = normalize(phi_centre)*220

    "upload phuz 2 slm"
    slm_disp_obj.display(phi_centre)

    figph = plt.figure()
    plt.imshow(phi_centre, cmap='inferno')
    plt.colorbar()
    plt.title("phi_centre")
    # plt.show()
    plt.show(block=False)
    plt.pause(1)
    plt.close(figph)


    "open shutter"
    sh.shutter_state()
    time.sleep(0.1)
    if sh.shut_state == 0:
        sh.shutter_enable()
    time.sleep(0.4)
    sh.shutter_state()

    # Take camera image
    cam_obj.prep_acq()
    cam_obj.take_image()
    imgzaz = cam_obj.last_frame

    plo_che = True
    if plo_che:
        fig = plt.figure()
        # plt.imshow(imgzaz, cmap='inferno')
        plt.imshow(imgzaz, cmap='inferno', vmax=5000)
        plt.colorbar()
        plt.title("full IMG")
        # plt.show()
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)

    " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ "
    " set roi or else "
    " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ "

    # cam_roi_pos = [1080, 1230]  # grat 10 [1230:1530, 1080:1380]
    cam_roi_pos = [874, 874]  # grat 10 [1230:1530, 1080:1380]
    cam_roi_sz = [300, 300]  # grat 10
    cam_obj.roi_set_roi(int(cam_roi_pos[0] * cam_obj.bin_sz), int(cam_roi_pos[1] * cam_obj.bin_sz),
                        int(cam_roi_sz[0] * cam_obj.bin_sz), int(cam_roi_sz[1] * cam_obj.bin_sz))

    cam_obj.stop_acq()
    cam_obj.exposure = 0.1/1000
    cam_obj.prep_acq()
    cam_obj.take_image()
    imgzaz = cam_obj.last_frame
    cam_obj.exposure = exp_time


    # Fit Gaussian to camera image
    p_opt, p_err = ft.fit_gaussian(imgzaz)
    calib_pos_x = int(p_opt[0] + cam_roi_sz[0] // 2)
    calib_pos_y = int(p_opt[1] + cam_roi_sz[1] // 2)
    print(Fore.LIGHTRED_EX + "ROi IMG, gaussian center x0 {}, "
                             "y0 {}".format(calib_pos_x, calib_pos_y) + Style.RESET_ALL)

    plo_che = True
    if plo_che:
        fig = plt.figure()
        plt.imshow(imgzaz, cmap='inferno', vmax=1500)  # grat 10
        # plt.imshow(imgzaz[1230:1530, 1080:1380], cmap='inferno', vmax=65000)  # grat 10
        plt.colorbar()
        plt.title("ROi IMG, gaussian center x0 {}, y0 {}".format(calib_pos_x, calib_pos_y))
        # plt.show()
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)
    #
    print(Fore.LIGHTGREEN_EX + "record background" + Style.RESET_ALL)
    # todo: need upload the proper phase for background acquisition, or wait a while for the camera to cool down
    # fixme: same goes for acquisition of first frame

    "close shutter"
    sh.shutter_state()
    time.sleep(0.4)
    if sh.shut_state == 1:
        sh.shutter_enable()
    time.sleep(0.4)
    sh.shutter_state()

    frame_num = 1
    cam_obj.stop_acq()

    # cam_obj.prep_acq()
    cam_obj.take_average_image(frame_num)
    cam_obj.bckgr = copy.deepcopy(cam_obj.last_frame)
    bckgr = copy.deepcopy(cam_obj.bckgr)

    if plo_che:
        fig = plt.figure()
        plt.imshow(bckgr, cmap='inferno', vmax=150)
        # plt.imshow(bckgr[1230:1530, 1080:1380], cmap='inferno', vmax=150)
        plt.colorbar()
        plt.title('backg')
        # plt.show()
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)

    "open shutter"
    sh.shutter_state()
    time.sleep(0.1)
    if sh.shut_state == 0:
        sh.shutter_enable()
    time.sleep(0.4)
    sh.shutter_state()

    # img = np.zeros((ny, nx, aperture_number ** 2))
    img = np.zeros((300, 300, aperture_number ** 2))

    aperture_power = np.zeros(aperture_number ** 2)
    slm_phase = phi_centre[:aperture_width, 124:124+aperture_width]

    plot_within = False
    dt = []

    print(Fore.LIGHTBLUE_EX + "cam_obj exposure is {}".format(cam_obj.exposure[0]) + Style.RESET_ALL)
    # todo: need upload the proper phase for background acquisition, or wait a while for the camera to cool down
    # fixme: same goes for acquisition of first frame, its very intense due to heating of the sensor by the full phase

    bckgr = np.copy(bckgr)
    iter_num = aperture_number ** 2
    for i in range(iter_num):
        print("iter {} of {}".format(i, aperture_number ** 2))
        t_start = time.time()
        masked_phase = np.zeros((npix, np.max(slm_disp_obj.res)))
        masked_phase[slm_idx[0][i]:slm_idx[1][i], slm_idx[2][i]:slm_idx[3][i]] = slm_phase

        slm_disp_obj.display(masked_phase)

        cam_obj.take_average_image(frame_num)
        img[..., i] = cam_obj.last_frame - bckgr

        aperture_power[i] = np.sum(img[..., i]) / (np.size(img[..., i]) * exp_time)
        # np.save(path + '\\imgF_iter_{}'.format(i), imgF)

        if plot_within:
            fig = plt.figure()
            plt.subplot(131), plt.imshow(img[..., i], cmap='inferno', vmin=0, vmax=1450)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('aperture_power[i]: {}'.format(aperture_power[i]))
            plt.subplot(132), plt.imshow(masked_phase, cmap='inferno')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('iter: {}'.format(i))
            plt.subplot(133), plt.imshow(img[..., i], cmap='inferno', vmax=600)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('ROI')
            plt.show(block=False)
            fig.savefig(path + '\\iter_{}'.format(i) + '_full.png', dpi=300, bbox_inches='tight', transparent=False)
            # Save data
            # np.save(path + '\\imgF_iter_{}'.format(i), imgF)
            plt.pause(0.8)
            plt.close(fig)

        dt.append(time.time() - t_start)
        print("time o iter: {}".format(dt[i]))
        print("iter {} of {}".format(i, iter_num))
        print("estimated time left approx: {}'".format((numpy.mean(dt)*(iter_num-i)) / 60))

    np.save(path + '//img', img)
    np.save(path + '//aperture_power', aperture_power)

    # Find SLM intensity profile
    i_rec = np.reshape(aperture_power, (aperture_number, aperture_number))
    # Save data
    np.save(path + '//i_rec', i_rec)

    figIint = plt.figure()
    plt.imshow(i_rec, cmap='inferno')
    plt.colorbar()
    plt.title("i_rec")
    # plt.show()
    plt.show(block=False)
    figIint.savefig(path + '\\intenseFull.png', dpi=300, bbox_inches='tight', transparent=False)
    # Save data
    # np.save(path + '\\imgF_iter_{}'.format(i), imgF)
    plt.pause(0.8)
    plt.close(figIint)

    # Fit Gaussian to measured intensity
    extent_slm = (slm_disp_obj.slm_size[0] + aperture_width * slm_disp_obj.pitch) / 2
    x_fit = np.linspace(-extent_slm, extent_slm, aperture_number)
    x_fit, y_fit = np.meshgrid(x_fit, x_fit)
    sig_x, sig_y = pms_obj.beam_diameter, pms_obj.beam_diameter
    popt_slm, perr_slm = ft.fit_gaussian(i_rec, dx=0, dy=0, sig_x=sig_x, sig_y=sig_y, xy=[x_fit, y_fit])

    i_fit_slm = pt.gaussian(slm_disp_obj.meshgrid_slm[0], slm_disp_obj.meshgrid_slm[1], *popt_slm)

    # Plotting
    extent_slm_mm = extent_slm * 1e3
    extent = [-extent_slm_mm, extent_slm_mm, -extent_slm_mm, extent_slm_mm]

    fig, axs = plt.subplots(1, 2)
    divider = make_axes_locatable(axs[0])
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    im = axs[0].imshow(i_rec / np.max(i_rec), cmap='turbo', extent=extent)
    axs[0].set_title('Intensity at SLM Aperture', fontname='Cambria')
    axs[0].set_xlabel("x [mm]", fontname='Cambria')
    axs[0].set_ylabel("y [mm]", fontname='Cambria')

    divider = make_axes_locatable(axs[1])
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    im = axs[1].imshow(i_fit_slm / np.max(i_fit_slm), cmap='turbo', extent=extent)
    axs[1].set_title('Fitted Gaussian', fontname='Cambria')
    axs[1].set_xlabel("x [mm]", fontname='Cambria')
    axs[1].set_ylabel("y [mm]", fontname='Cambria')
    cbar = plt.colorbar(im, cax=ax_cb)
    cbar.set_label('normalised intensity', fontname='Cambria')
    # plt.show()
    plt.show(block=False)
    fig.savefig(path + '\\intense.png', dpi=300, bbox_inches='tight', transparent=False)
    # Save data
    # np.save(path + '\\imgF_iter_{}'.format(i), imgF)
    plt.pause(1.8)
    plt.close(fig)

    figSub = plt.figure()
    plt.imshow(img[..., (aperture_number ** 2 - aperture_number) // 2], cmap='turbo')
    plt.title('Camera image of central sub-aperture')
    # plt.show()
    plt.show(block=False)
    figSub.savefig(path + '\\central_sub.png', dpi=300, bbox_inches='tight', transparent=False)
    # Save data
    # np.save(path + '\\imgF_iter_{}'.format(i), imgF)
    plt.pause(1.8)
    plt.close(figSub)

    # Save data
    # np.save(path + '//i_rec', i_rec)
    np.save(path + '//i_fit_slm', i_fit_slm)
    np.save(path + '//popt_slm', popt_slm)
    return path + '//i_rec'


def measure_slm_wavefront(slm_disp_obj, cam_obj, pms_obj, aperture_number, aperture_width, img_size, exp_time, spot_pos,
                          n_avg_frames=10, benchmark=False, phi_load_path=None, roi_min_x=16, roi_min_y=16, roi_n=8):
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
    os.mkdir(path)

    "the benchmark situation"
    # todo: no eye deer how zis works, have a lOOk
    print(Fore.RED + "benchmark: {}".format(benchmark) + Style.RESET_ALL)
    if benchmark is True:
        phi_load = np.load(phi_load_path)
    else:
        phi_load = np.zeros((aperture_number, aperture_number))

    "set res & stuff"
    # roi_mem = cam_obj.roi
    res_y, res_x = slm_disp_obj.res
    npix = np.min(slm_disp_obj.res)
    border_x = int(np.abs(res_x - res_y) // 2)
    zeros = np.zeros((npix, npix))
    zeros_full = np.zeros((npix, np.max(slm_disp_obj.res)))
    fl = pms_obj.fl

    "their phase, diagonal one"
    # lin_phase = np.array([-spot_pos, -spot_pos])
    # slm_phase4me = pt.init_phase(np.zeros((1024, 1272)), slm_disp_obj,
    #                              pms_obj, lin_phase=lin_phase)
    # slm_phaseOUT = pt.init_phase(np.zeros((aperture_width, aperture_width)), slm_disp_obj,
    #                              pms_obj, lin_phase=lin_phase)

    # slm_phase4me = np.remainder(slm_phase4me, 2 * np.pi)
    # slm_phase4me = np.fliplr(slm_phase4me)
    # slm_phaseNOR = normalize(slm_phase4me)

    phuzGen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": True}
    phuzGen.linear_grating()
    # phuzGen.grat = slm_phaseNOR
    phuzGen._make_full_slm_array()
    slm_phase = phuzGen.final_phuz
    # phi_centre = normalize(phi_centre)*220

    figph = plt.figure()
    plt.imshow(slm_phase, cmap='inferno')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("slm_phase")
    # plt.show()
    plt.show(block=False)
    plt.pause(0.6)
    plt.close(figph)

    slm_idx = get_aperture_indices(aperture_number, aperture_number, border_x, npix + border_x - 1, 0,
                                   npix - 1, aperture_width, aperture_width)

    n_centre = aperture_number ** 2 // 2 + aperture_number // 2 - 1
    n_centre_ref = aperture_number ** 2 // 2 + aperture_number // 2

    idx = range(aperture_number ** 2)
    idx = np.reshape(idx, (aperture_number, aperture_number))
    idx = idx[roi_min_x:roi_min_x + roi_n, roi_min_y:roi_min_y + roi_n]
    idx = idx.flatten()

    phi_int = np.zeros_like(zeros_full)

    "phase 4 check"
    phi_int[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]] = \
        slm_phase[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]]
    phi_int[slm_idx[0][n_centre_ref]:slm_idx[1][n_centre_ref], slm_idx[2][n_centre_ref]:slm_idx[3][n_centre_ref]] = \
        slm_phase[slm_idx[0][n_centre_ref]:slm_idx[1][n_centre_ref], slm_idx[2][n_centre_ref]:slm_idx[3][n_centre_ref]]

    "upload 2 slm"
    slm_disp_obj.display(phi_int)

    "Load measured laser intensity profile"
    laser_intensity_measured = np.load(pms_obj.i_path)
    laser_intensity_upscaled = pt.load_filter_upscale(laser_intensity_measured, npix, 1,
                                                      filter_size=pms_obj.i_filter_size)
    laser_intensity_upscaled = np.pad(laser_intensity_upscaled, ((0, 0), (border_x, border_x)))

    "roi"
    # cam_roi_pos = [1080, 1230]  # grat 10 [1230:1530, 1080:1380]
    cam_roi_pos = [874, 874]  # grat 10 [1230:1530, 1080:1380]
    cam_roi_sz = [300, 300]  # grat 10
    cam_obj.roi_set_roi(int(cam_roi_pos[0] * cam_obj.bin_sz), int(cam_roi_pos[1] * cam_obj.bin_sz),
                        int(cam_roi_sz[0] * cam_obj.bin_sz), int(cam_roi_sz[1] * cam_obj.bin_sz))


    cam_obj.prep_acq()
    cam_obj.take_image()
    imgzaz = cam_obj.last_frame

    img_size = imgzaz.shape[0]

    plo_che = True
    if plo_che:
        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(imgzaz, cmap='inferno', vmax=400)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('full IMG max V: {}'.format(np.amax(imgzaz)))
        plt.subplot(132)
        # plt.imshow(imgzaz[1230:1530, 1080:1380], cmap='inferno', vmax=400)
        plt.imshow(imgzaz, cmap='inferno', vmax=400)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("ROi IMG")
        plt.subplot(133)
        plt.imshow(phi_int, cmap='inferno')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("phi_int")
        # plt.show()
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)

    # Take camera images
    p_max = np.sum(laser_intensity_upscaled[slm_idx[0][n_centre]:slm_idx[1][n_centre],
                   slm_idx[2][n_centre]:slm_idx[3][n_centre]])

    norm = np.zeros(roi_n ** 2)
    img = np.zeros((300, 300, roi_n ** 2))
    # img = np.zeros((img_size, img_size, roi_n ** 2))
    aperture_power = np.zeros(roi_n ** 2)
    aperture_powah = np.zeros(roi_n ** 2)
    dt = []
    aperture_width_adj = np.zeros(roi_n ** 2)

    print(Fore.LIGHTGREEN_EX + "record background" + Style.RESET_ALL)

    "close shutter"
    sh.shutter_state()
    time.sleep(0.4)
    if sh.shut_state == 1:
        sh.shutter_enable()
    time.sleep(0.4)
    sh.shutter_state()

    frame_num = n_avg_frames
    cam_obj.stop_acq()
    cam_obj.take_average_image(frame_num)
    cam_obj.bckgr = copy.deepcopy(cam_obj.last_frame)
    print(cam_obj.bckgr.shape)
    bckgr = copy.deepcopy(cam_obj.bckgr)

    if plo_che:
        fig = plt.figure()
        plt.imshow(bckgr, cmap='inferno', vmax=150)
        # plt.imshow(bckgr[1230:1530, 1080:1380], cmap='inferno', vmax=150)
        plt.colorbar(fraction=0.046, pad=0.04)
        # plt.colorbar()
        plt.title('backg')
        # plt.show()
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)

    "open shutter"
    sh.shutter_state()
    time.sleep(0.1)
    if sh.shut_state == 0:
        sh.shutter_enable()
    time.sleep(0.4)
    sh.shutter_state()

    'main lOOp'
    plot_within = False
    aperture_coverage = np.copy(zeros_full)
    iter_num = roi_n ** 2
    for i in range(iter_num):
        # i = 465
        t_start = time.time()
        ii = idx[i]
        idx_0, idx_1 = np.unravel_index(ii, phi_load.shape)

        norm[i] = p_max / np.sum(laser_intensity_upscaled[slm_idx[0][ii]:slm_idx[1][ii], slm_idx[2][ii]:slm_idx[3][ii]])
        masked_phase = np.copy(zeros_full)
        aperture_coverage_now = np.copy(zeros_full)
        aperture_width_tar = np.sqrt(aperture_width ** 2 * norm[i])
        pad = int((aperture_width_tar - aperture_width) // 2)
        aperture_width_adj[i] = aperture_width + 2 * pad
        # pad = 0  # fixme: what 2 do with ziz

        masked_phase[slm_idx[0][ii] - pad:slm_idx[1][ii] + pad, slm_idx[2][ii] - pad:slm_idx[3][ii] + pad] = \
            slm_phase[slm_idx[0][ii] - pad:slm_idx[1][ii] + pad, slm_idx[2][ii] - pad:slm_idx[3][ii] + pad]
        masked_phase[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]] = \
            slm_phase[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]]

        aperture_coverage_now[slm_idx[0][ii] - pad:slm_idx[1][ii] + pad, slm_idx[2][ii] - pad:slm_idx[3][ii] + pad] = 1
        aperture_coverage += aperture_coverage_now

        phiphiphi = np.remainder(masked_phase, 2 * np.pi)

        phuzGen.grat = phiphiphi
        phuzGen._make_full_slm_array()
        # phi_pou = phuzGen.final_phuz
        # phi_rou = normalize(phi_pou) * 220

        slm_disp_obj.display(masked_phase)
        if i == 0:
            time.sleep(2 * 0.1)

        cam_obj.take_average_image(n_avg_frames)
        img_avg = cam_obj.last_frame - bckgr
        img[:, :, i] = cam_obj.last_frame - bckgr
        aperture_power[i] = np.mean(img[..., i][142:144, 152:154]) * 3 ** 2 / 3 ** 2
        # print("aperture_power[i]: {}".format(aperture_power[i]))
        aperture_powah[i] = img[..., i][143, 153]
        # print("pxl_power[i]: {}".format(aperture_powah[i]))

        # np.save(path + '\\imgF_iter_{}'.format(i), img[:, :, i])
        # np.save(path + '\\masked_phase_iter_{}'.format(i), masked_phase)
        if plot_within:
            fig = plt.figure()
            plt.subplot(131), plt.imshow(img_avg, cmap='inferno', vmin=0, vmax=40)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('max V: {}'.format(np.amax(img_avg)))
            plt.subplot(132), plt.imshow(masked_phase, cmap='inferno')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('iter: {}, pad is {}'.format(i, pad))
            plt.subplot(133), plt.imshow(img[..., i], cmap='inferno', vmin=0, vmax=200)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('ROI ape_pow: {}, pxl_pow: {}'.format(np.round(aperture_power[i],2),
                                                            np.round(aperture_powah[i], 2)))
            # plt.show()
            # plt.show(block=False)
            fig.savefig(path + '\\iter_{}'.format(i) + '_phuz.png', dpi=300, bbox_inches='tight',
                        transparent=False)  # True trns worls nice for dispersion thinks I
            # Save data
            # np.save(path + '\\imgF_iter_{}'.format(i), imgF)
            # np.save(path + '\\imgF_iter_{}'.format(i), img[:, :, i])
            # np.save(path + '\\masked_phase_iter_{}'.format(i), masked_phase)
            plt.pause(0.4)
            plt.close(fig)

        dt.append(time.time() - t_start)
        print("time o iter: {}".format(dt[i]))
        print("iter {} of {}".format(i, iter_num))
        print("estimated time left approx: {}'".format((numpy.mean(dt)*(iter_num-i)) / 60))
    t = np.cumsum(dt)

    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    "~~~ lOOp enD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    "plots & saves"
    np.save(path + '/img', img)
    saVe_plo = True

    figu = plt.figure()
    plt.imshow(aperture_coverage)
    plt.title('Coverage of sub-apertures on the SLM')
    if saVe_plo:
        plt.show(block=False)
        figu.savefig(path +'\\Coverage.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.8)
        plt.close()
    else:
        plt.show()

    # Fit sine to images
    fit_sine = ft.FitSine(fl, pms_obj.k)

    popt_sv = []
    pcov_sv = []
    perr_sv = []

    x, y = pt.make_grid(img[:, :, 0], scale=cam_obj.pitch)

    print(Fore.LIGHTBLACK_EX + "start em fitz" + Style.RESET_ALL)

    "fit dem d@a"
    for i in range(roi_n ** 2):
        ii = idx[i]
        img_i = img[:, :, i]

        dx = (slm_idx[2][ii] - slm_idx[2][n_centre]) * slm_disp_obj.pitch
        dy = (slm_idx[0][ii] - slm_idx[0][n_centre]) * slm_disp_obj.pitch
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

    # Save data
    np.save(path + '//dphi', dphi)
    np.save(path + '//dphi_uw', dphi_uw)
    np.save(path + '//dphi_err', dphi_err)
    np.save(path + '//dx', dx)
    np.save(path + '//dy', dy)
    np.save(path + '//i_fit', i_fit)
    np.save(path + '//dphi_uw_mask', dphi_uw_mask)
    np.save(path + '//i_fit_mask', i_fit_mask)
    np.save(path + '//t', t)
    np.save(path + '//popt_sv', popt_sv)
    np.save(path + '//perr_sv', perr_sv)
    np.save(path + '//aperture_coverage', aperture_coverage)
    # np.save(path + '//cal_pos_y', cal_pos_y)


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

    power = np.reshape(aperture_power, (roi_n, roi_n))
    powah = np.reshape(aperture_powah, (roi_n, roi_n))
    np.save(path + '//powah', powah)
    np.save(path + '//power', power)
    np.save(path + '//imgzz', img)

    figPow = plt.Figure()
    plt.subplot(121)
    plt.imshow(powah, cmap='turbo')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(122)
    plt.imshow(power, cmap='turbo')
    plt.colorbar(fraction=0.046, pad=0.04)
    if saVe_plo:
        plt.show(block=False)
        figPow.savefig(path +'\\powaher.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(2)
        plt.close(figPow)
    else:
        plt.show()

    return path + '//dphi_uw'

# es el finAl
