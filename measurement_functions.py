"""
Module to measure the constant amplitude and phase at the SLM.
"""

import os
import time
import numpy as np
import hardware as hw
import error_metrics as m, patterns as pt, fitting as ft
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from peripheral_instruments.thorlabs_shutter import shutter as sh
from colorama import Fore, Style  # , Back

import slm.helpers
from slm.phase_generator import phagen as phuzGen
import copy

def normalize(im):
    """normalizez 0 2 1"""
    maxi = np.max(im)
    mini = np.min(im)
    norm = ((im - mini) / (maxi - mini))
    return norm


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


def init_pha(img, slm_disp_large_dim, slm_disp_pitch, pms_obj, lin_phase=None, quad_phase=None, lin_method=None):
    """
    SLM phase guess to initialise phase-retrieval algorithm (see https://doi.org/10.1364/OE.16.002176).

    :param ndarray img: 2D array with size of desired output.
    :param slm_disp_obj: Instance of Params class
    :param ndarray lin_phase: Vector of length 2, containing parameters for the linear phase term
    :param ndarray quad_phase: Vector of length 2, containing parameters for the quadratic phase term
    :param str lin_method: Determines how the linear phase term is parameterised. The options are:

        -'pixel'
            Defines the linear phase in terms of Fourier pixels [px].
        -'angles'
            Defines the linear phase in terms of angles [rad].

    :return: Phase pattern of shape ``img.shape``
    """
    if lin_phase is None:
        lin_phase = np.zeros(2)
    if lin_method is None:
        lin_method = 'pixel'
    if quad_phase is None:
        quad_phase = np.zeros(2)

    x, y = make_grid(img)

    if lin_method == 'pixel':
        pix_x, pix_y = lin_phase
        mx = np.pi * pix_x / slm_disp_large_dim
        my = np.pi * pix_y / slm_disp_large_dim
        print('pixel methOD')
    if lin_method == 'angles':
        alpha_x, alpha_y = lin_phase
        mx = np.tan(alpha_x) * pms_obj.k * slm_disp_pitch
        my = np.tan(alpha_y) * pms_obj.k * slm_disp_pitch

    r, gamma = quad_phase
    kl = mx * x + my * y
    kq = 4 * r * (gamma * y ** 2 + (1 - gamma) * x ** 2)
    return kl + kq


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
    # roi_mem = cam_obj.roi_is
    date_saved = time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())
    path = pms_obj.data_path + date_saved + '_' + 'measure_slm_intensity'
    # os.mkdir(path)

    res_y, res_x = 1024, 1272
    border_x = int(np.abs(res_x - res_y) // 2)
    npix = 1024

    zeros = np.zeros((npix, npix))
    zeros_full = np.zeros((npix, np.max(1272)))

    lin_phase = np.array([-spot_pos, -spot_pos])
    slm_phase4me = init_pha(np.zeros((1024, 1024)), 1272, 12.5e-6,
                         pms_obj, lin_phase=lin_phase)
    slm_phaseOUT = init_pha(np.zeros((aperture_width, aperture_width)), 1272, 12.5e-6,
                         pms_obj, lin_phase=lin_phase)

    # slm_phase = pt.init_phase(np.zeros((aperture_width, aperture_width)), slm_disp_obj, pms_obj, lin_phase=lin_phase)
    slm_phase = np.remainder(slm_phaseOUT, 2 * np.pi)
    slm_phase4me = np.remainder(slm_phase4me, 2 * np.pi)
    slm_phase4me = np.fliplr(slm_phase4me)

    # slm_phase = slm_phase
    # slm_phase = np.flipud(np.fliplr(slm_phase))
    slm_idx = get_aperture_indices(aperture_number, aperture_number, border_x, npix + border_x, 0, npix, aperture_width,
                                   aperture_width)

    # Display central sub-aperture on SLM and check if camera is over-exposed.
    i = (aperture_number ** 2) // 2 - aperture_number // 2
    phi_centre = np.zeros_like(zeros)
    phi_centre[slm_idx[0][i]:slm_idx[1][i], slm_idx[2][i]:slm_idx[3][i]] = slm_phase
    # phi_centre = slm_phase


    phuzGen.diviX = 10
    phuzGen.diviY = 10
    phuzGen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": True}
    phuzGen.linear_grating()
    phi_centreMA = phuzGen.final_phuz

    # figph = plt.figure()
    # plt.imshow(phi_centre, cmap='inferno')
    # plt.colorbar()
    # plt.title("phi_centre")
    # plt.show()

    slm_phaseNOR = normalize(slm_phase)*220
    phi_centreMA = normalize(phi_centreMA)*220
    phi_centreMA = phi_centreMA[:aperture_width, :aperture_width]

    plo_che = True
    if plo_che:
        fig = plt.figure()
        plt.subplot(221), plt.imshow(slm_phase4me, cmap='inferno')
        plt.colorbar()
        plt.title('slm_phase4me, theirs normalized')
        plt.subplot(222), plt.imshow(slm_phase, cmap='inferno')
        plt.colorbar()
        plt.title('slm_phase')
        plt.subplot(223), plt.imshow(phi_centre, cmap='inferno')
        plt.colorbar()
        plt.title('phi_centre')
        plt.subplot(224), plt.imshow(phi_centreMA, cmap='inferno')
        plt.colorbar()
        plt.title('phi_centreMAap')
        plt.show()
        # plt.show(block=False)
        # fig.savefig(path + '\\iter_{}'.format(i) + '_full.png', dpi=300, bbox_inches='tight',
        #             transparent=False)  # True trns worls nice for dispersion thinks I
        # plt.pause(0.8)
        # plt.close(fig)




    # measure_slm_intensity.img_exp_check = cam_obj.get_image(exp_time)
    # measure_slm_intensity.img_exp_check = cam_obj.take_image()

    # # Find Camera position with respect to SLM
    # popt_clb, img_cal = find_camera_position(slm_disp_obj, cam_obj, pms_obj, lin_phase, exp_time=exp_time / 10,
    #                                          aperture_diameter=npix // 20, roi=[400, 400])
    #
    # lin_phase = np.array([-spot_pos, -spot_pos])
    # slm_phase = pt.init_phase(np.zeros((aperture_width, aperture_width)), slm_disp_obj, pms_obj, lin_phase=lin_phase)
    # # slm_phase = np.remainder(slm_phase, 2 * np.pi)
    # slm_phase = np.flipud(np.fliplr(slm_phase))
    #

    # # ny, nx = cam_obj.res
    # ny, nx = img_cal.shape
    # calib_pos_x = int(popt_clb[0] + nx // 2)
    # calib_pos_y = int(popt_clb[1] + ny // 2)

    # plt.figure()
    # plt.imshow(img_cal, cmap='inferno', vmax=1000)
    # plt.plot(calib_pos_x, calib_pos_y, 'wx', color='g')
    # plt.title('Camera image and fitted spot position')
    # plt.show()

    # Take camera images
    # print("roi_width, {}".format(roi_width))
    # print("int((calib_pos_y - roi_width / 2) // 2 * 2), {}".format(int((calib_pos_y - roi_width / 2) // 2 * 2)))
    # print("int((calib_pos_x - roi_width / 2) // 2 * 2), {}".format(int((calib_pos_x - roi_width / 2) // 2 * 2)))
    # roi = [roi_width, roi_width, int((calib_pos_x - roi_width / 2) // 2 * 2),
    #        int((calib_pos_y - roi_width / 2) // 2 * 2)]
    # cam_obj.roi = roi

    # cam_obj.start(aperture_number ** 2)
    # cam_obj.num = aperture_number ** 2
    # cam_obj.num = aperture_number
    # cam_obj.hcam.setACQMode('fixed_length', number_frames=cam_obj.num)

    print(Fore.LIGHTGREEN_EX + "record background" + Style.RESET_ALL)


    # img = np.zeros((ny, nx, aperture_number ** 2))

    img = np.zeros((90, 90, aperture_number ** 2))
    # img = np.zeros((bckgr.shape[0], bckgr.shape[1], aperture_number ** 2))

    # img = np.zeros((roi[1], roi[0], aperture_number ** 2))
    aperture_power = np.zeros(aperture_number ** 2)
    # slm_phase = normalize(slm_phase)*200phi_centre
    slm_phase = phi_centre[:aperture_width, :aperture_width]
    plot_within = True
    # here = [108, 109, 110]
    # here = [434, 435, 436]
    dt = np.zeros(aperture_number ** 2)

    # for i in here:  # range(aperture_number ** 2):
    for i in range(aperture_number ** 2):
        # i = (aperture_number ** 2) // 2 - aperture_number // 2
        # i = i-3
        print("iter {} of {}".format(i, aperture_number ** 2))
        t_start = time.time()
        # masked_phase = np.copy(zeros_full)
        masked_phase = np.zeros((npix, np.max(slm_disp_obj.res)))
        masked_phase[slm_idx[0][i]:slm_idx[1][i], slm_idx[2][i]:slm_idx[3][i]] = slm_phase

        # img[..., i] = cam_obj.last_frame
        # plt.imshow(imgzaz[1120:1210, 1380:1470], cmap='inferno', vmax=65000)

        aperture_power[i] = np.sum(img[..., i]) / (np.size(img[..., i]) * exp_time)
        print(aperture_power[i])

        if plot_within:
            fig = plt.figure()
            plt.subplot(221), plt.imshow(masked_phase, cmap='inferno', vmax=150)
            plt.colorbar()
            plt.title('aperture_power[i]: {}'.format(aperture_power[i]))
            plt.subplot(222), plt.imshow(masked_phase, cmap='inferno')
            plt.colorbar()
            plt.title('iter: {}'.format(i))
            plt.subplot(223), plt.imshow(img[..., i], cmap='inferno', vmax=150)
            plt.colorbar()
            plt.title('ROI')
            plt.subplot(224), plt.imshow(masked_phase[1120:1210, 1380:1470], cmap='inferno', vmax=150)
            plt.colorbar()
            plt.title('bg')
            # plt.show()
            plt.show(block=False)
            # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
            fig.savefig(path + '\\iter_{}'.format(i) + '_full.png', dpi=300, bbox_inches='tight',
                        transparent=False)  # True trns worls nice for dispersion thinks I
            plt.pause(0.8)
            plt.close(fig)

        dt[i] = time.time() - t_start
        print("time of iter: {}".format(dt[i]))
    # cam_obj.stop()
    # cam_obj.roi = roi_mem

    np.save(path + '//img', img)
    np.save(path + '//aperture_power', aperture_power)

    # Find SLM intensity profile
    i_rec = np.reshape(aperture_power, (aperture_number, aperture_number))
    # Save data
    np.save(path + '//i_rec', i_rec)

    fig = plt.figure()
    plt.imshow(i_rec, cmap='inferno')
    plt.colorbar()
    plt.title("i_rec")
    plt.show()

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
    plt.show()

    plt.figure()
    plt.imshow(img[..., (aperture_number ** 2 - aperture_number) // 2], cmap='turbo')
    plt.title('Camera image of central sub-aperture')
    plt.show()

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
    
    # prep data Save
    date_saved = time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())
    path = pms_obj.data_path + date_saved + '_measure_slm_wavefront'
    os.mkdir(path)

    # roi_mem = cam_obj.roi
    res_y, res_x = 1024, 1272
    npix = np.min(1024)
    border_x = int(np.abs(res_x - res_y) // 2)
    zeros_full = np.zeros((npix, 1272))
    #
    fl = pms_obj.fl
    #
    lin_phase = np.array([-spot_pos, -spot_pos])

    slm_phase4me = init_pha(np.zeros((1024, 1272)), 1272, 12.5e-6,
                         pms_obj, lin_phase=lin_phase)
    slm_phaseOUT = init_pha(np.zeros((aperture_width, aperture_width)), 1272, 12.5e-6,
                         pms_obj, lin_phase=lin_phase)

    # slm_phase = pt.init_phase(np.zeros((aperture_width, aperture_width)), slm_disp_obj, pms_obj, lin_phase=lin_phase)
    slm_phase = np.remainder(slm_phaseOUT, 2 * np.pi)
    slm_phase4me = np.remainder(slm_phase4me, 2 * np.pi)
    slm_phase4me = np.fliplr(slm_phase4me)
    # slm_phase = np.remainder(slm_phase, 2 * np.pi)
    slm_phase = slm_phase4me
    # slm_phase = np.flipud(np.fliplr(slm_phase))
    #
    if benchmark is True:
        phi_load = np.load(phi_load_path)
    else:
        phi_load = np.zeros((aperture_number, aperture_number))

    slm_phaseNOR = normalize(slm_phase)
    phuzGen.diviX = 10
    phuzGen.diviY = 10
    phuzGen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": True}
    # phuzGen.linear_grating()
    phuzGen.grat = slm_phaseNOR
    phuzGen._make_full_slm_array()
    phi_centre = phuzGen.final_phuz

    # # figph = plt.figure()
    # plt.imshow(phi_centre, cmap='inferno')
    # plt.colorbar()
    # plt.title("phi_centre")
    # plt.show()

    phi_centre = normalize(phi_centre)*220
    # slm_disp_obj.display(phi_centre)
    # slm_phase = phi_centre[:aperture_width, :aperture_width]
    slm_phase = phi_centre

    slm_idx = get_aperture_indices(aperture_number, aperture_number, border_x, npix + border_x - 1, 0,
                                   npix - 1, aperture_width, aperture_width)

    n_centre = aperture_number ** 2 // 2 + aperture_number // 2 - 1
    n_centre_ref = aperture_number ** 2 // 2 + aperture_number // 2

    idx = range(aperture_number ** 2)
    idx = np.reshape(idx, (aperture_number, aperture_number))
    idx = idx[roi_min_x:roi_min_x + roi_n, roi_min_y:roi_min_y + roi_n]
    idx = idx.flatten()

    phi_int = np.zeros_like(zeros_full)
    # print('slm_idx {}'.format(slm_idx))
    # print('n_centre {}'.format(n_centre))
    # print('slm_idx[0][n_centre]: {}, slm_idx[1][n_centre]: {}, '
    #       'slm_idx[2][n_centre]: {}, slm_idx[3][n_centre]: {}'.format(slm_idx[0][n_centre], slm_idx[1][n_centre],
    #                                         slm_idx[2][n_centre], slm_idx[3][n_centre]))

    phi_int[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]] = \
        slm_phase[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]]
    phi_int[slm_idx[0][n_centre_ref]:slm_idx[1][n_centre_ref], slm_idx[2][n_centre_ref]:slm_idx[3][n_centre_ref]] = \
        slm_phase[slm_idx[0][n_centre_ref]:slm_idx[1][n_centre_ref], slm_idx[2][n_centre_ref]:slm_idx[3][n_centre_ref]]

    plt.imshow(phi_int, cmap='inferno')  # grat 10
    plt.colorbar()
    plt.title("n_centre {}, n_centre_ref {}".format(n_centre, n_centre_ref))
    plt.show()

    # # cam_obj.start()
    # measure_slm_wavefront.img_exposure_check = cam_obj.get_image(exp_time)
    # # cam_obj.stop()

    # Load measured laser intensity profile
    laser_intensity_measured = np.load(pms_obj.i_path)
    laser_intensity_upscaled = pt.load_filter_upscale(laser_intensity_measured, npix, 1,
                                                      filter_size=pms_obj.i_filter_size)
    laser_intensity_upscaled = np.pad(laser_intensity_upscaled, ((0, 0), (border_x, border_x)))

    # # Find Camera position with respect to SLM
    # popt_clb, img_cal = find_camera_position(slm_disp_obj, cam_obj, pms_obj, lin_phase, exp_time=exp_time / 20,
    #                                          aperture_diameter=npix // 20, roi=[400, 400])
    #
    # cal_pos_x = popt_clb[0] + cam_obj.res[1] // 2
    # cal_pos_y = popt_clb[1] + cam_obj.res[0] // 2
    #
    # plt.figure()
    # plt.imshow(img_cal, cmap='turbo')
    # plt.plot(cal_pos_x, cal_pos_y, 'wx')
    #
    # # Determine region of interest on camera
    # w_cam = int(img_size) // 2 * 2
    # h_cam = int(img_size) // 2 * 2
    # offset_x = int((cal_pos_x - w_cam // 2) // 2 * 2)
    # offset_y = int((cal_pos_y - h_cam // 2) // 2 * 2)
    # roi_cam = [w_cam, h_cam, offset_x, offset_y]

    # cam_roi_pos = [970, 590]  # grat 10
    # cam_roi_sz = [350, 350]  # grat 10
    cam_roi_pos = [1170, 804]  # grat 20
    cam_roi_sz = [220, 220]  # grat 20
    cam_obj.roi_set_roi(int(cam_roi_pos[0] * cam_obj.bin_sz), int(cam_roi_pos[1] * cam_obj.bin_sz),
                        int(cam_roi_sz[0] * cam_obj.bin_sz), int(cam_roi_sz[1] * cam_obj.bin_sz))

    cam_obj.prep_acq()
    cam_obj.take_image()
    imgzaz = cam_obj.last_frame

    img_size = imgzaz.shape[0]

    plo_che = True
    if plo_che:
        fig = plt.figure()
        plt.imshow(imgzaz, cmap='inferno', vmax=150)
        plt.colorbar()
        plt.title("ROi IMG")
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)

    # Take camera images
    p_max = np.sum(laser_intensity_upscaled[slm_idx[0][n_centre]:slm_idx[1][n_centre],
                   slm_idx[2][n_centre]:slm_idx[3][n_centre]])

    norm = np.zeros(roi_n ** 2)
    img = np.zeros((img_size, img_size, roi_n ** 2))
    aperture_power = np.zeros(roi_n ** 2)
    dt = np.zeros(roi_n ** 2)
    aperture_width_adj = np.zeros(roi_n ** 2)

    # n_img = int(2 * n_avg_frames * roi_n ** 2)
    # cam_obj.roi = roi_cam
    # cam_obj.start(n_img)

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
    # print(bckgr.shape)

    if plo_che:
        fig = plt.figure()
        plt.imshow(bckgr, cmap='inferno', vmax=150)
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

    'main lOOp'
    plot_within = False
    aperture_coverage = np.copy(zeros_full)
    for i in range(roi_n ** 2):
        # i = 64
        t_start = time.time()
        ii = idx[i]
        idx_0, idx_1 = np.unravel_index(ii, phi_load.shape)

        norm[i] = p_max / np.sum(laser_intensity_upscaled[slm_idx[0][ii]:slm_idx[1][ii], slm_idx[2][ii]:slm_idx[3][ii]])
        masked_phase = np.copy(zeros_full)
        aperture_coverage_now = np.copy(zeros_full)
        aperture_width_tar = np.sqrt(aperture_width ** 2 * norm[i])
        pad = int((aperture_width_tar - aperture_width) // 2)
        aperture_width_adj[i] = aperture_width + 2 * pad

        masked_phase[slm_idx[0][ii] - pad:slm_idx[1][ii] + pad, slm_idx[2][ii] - pad:slm_idx[3][ii] + pad] = \
            slm_phase[slm_idx[0][ii] - pad:slm_idx[1][ii] + pad, slm_idx[2][ii] - pad:slm_idx[3][ii] + pad] + \
            phi_load[idx_0, idx_1]
        masked_phase[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]] = \
            slm_phase[slm_idx[0][n_centre]:slm_idx[1][n_centre], slm_idx[2][n_centre]:slm_idx[3][n_centre]]

        aperture_coverage_now[slm_idx[0][ii] - pad:slm_idx[1][ii] + pad, slm_idx[2][ii] - pad:slm_idx[3][ii] + pad] = 1
        aperture_coverage += aperture_coverage_now

        slm_disp_obj.display(np.remainder(masked_phase, 2 * np.pi))
        if i == 0:
            # time.sleep(2 * slm_disp_obj.delay)
            time.sleep(2 * 0.1)

        # img_avg = hw.get_image_avg(cam_obj, exp_time, n_avg_frames)  # all as usual
        cam_obj.take_average_image(n_avg_frames)
        # img_avg = cam_obj.last_frame
        img_avg = cam_obj.last_frame - bckgr

        img[:, :, i] = np.copy(img_avg)
        aperture_power[i] = np.mean(img[:, :, i]) * aperture_width ** 2 / aperture_width_adj[i] ** 2
        print("aperture_power[i]: {}".format(aperture_power[i]))

        if plot_within:
            fig = plt.figure()
            plt.subplot(121), plt.imshow(img[..., i], cmap='inferno', vmax=50)
            plt.colorbar()
            plt.title('aperture_power[i]: {}'.format(aperture_power[i]))
            plt.subplot(122), plt.imshow(masked_phase, cmap='inferno')
            plt.colorbar()
            plt.title('aperture_power[i]: {}'.format(aperture_power[i]))
            # plt.show()
            plt.show(block=False)
            plt.pause(0.8)
            plt.close(fig)


        dt[i] = time.time() - t_start
        # print(dt[i])
        # print(i)
        print("time o iter: {}".format(dt[i]))
        print("iter {} of {}".format(i, roi_n ** 2))
    # cam_obj.stop()
    # cam_obj.roi = roi_mem
    t = np.cumsum(dt)

    # Save data
    date_saved = time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())
    path = pms_obj.data_path + date_saved + '_measure_slm_wavefront'
    os.mkdir(path)
    np.save(path + '/img', img)

    saVe_plo = True

    figu = plt.figure()
    plt.imshow(aperture_coverage)
    plt.title('Coverage of sub-apertures on the SLM')
    if saVe_plo:
        plt.show(block=False)
        # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
        figu.savefig(path +'\\Coverage.png', dpi=300, bbox_inches='tight',
                    transparent=False)  # True trns worls nice for dispersion thinks I
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

    phFig = plt.figure()
    plt.imshow(dphi_uw / np.pi / 2, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Unwrapped measured phase')
    if saVe_plo:
        plt.show(block=False)
        # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
        phFig.savefig(path +'\\Unwrapped.png', dpi=300, bbox_inches='tight',
                    transparent=False)  # True trns worls nice for dispersion thinks I
        plt.pause(0.8)
        plt.close()
    else:
        plt.show()

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
        fig1.savefig(path +'\\fit.png', dpi=300, bbox_inches='tight',
                    transparent=False)  # True trns worls nice for dispersion thinks I
        plt.pause(0.8)
        plt.close()
    else:
        plt.show()

    # Save data
    np.save(path + '//dphi', dphi)
    np.save(path + '//dphi_uw', dphi_uw)
    # np.save(path + '//cal_pos_x', cal_pos_x)
    # np.save(path + '//cal_pos_y', cal_pos_y)
    np.save(path + '//i_fit', i_fit)
    np.save(path + '//dphi_uw_mask', dphi_uw_mask)
    np.save(path + '//i_fit_mask', i_fit_mask)
    np.save(path + '//t', t)
    np.save(path + '//popt_sv', popt_sv)
    np.save(path + '//perr_sv', perr_sv)
    return path + '//dphi_uw'


def way_of_the_lens(slm_disp_obj, cam_obj, pms_obj, aperture_number, aperture_width, exp_time, spot_pos, roi_width):
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
    # roi_mem = cam_obj.roi_is
    date_saved = time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())
    path = pms_obj.data_path + '_' + 'back2thePrimitive'
    # path = pms_obj.data_path + date_saved + '_' + 'measure_slm_intensity'
    if not os.path.exists(path):
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
    i = (aperture_number ** 2) // 2 - aperture_number // 2
    phi_centre = np.zeros_like(zeros)
    # phi_centre[slm_idx[0][i]:slm_idx[1][i], slm_idx[2][i]:slm_idx[3][i]] = slm_phase
    phi_centre = slm_phase
    slm_phaseNOR = normalize(slm_phase)

    # plt.imshow(slm_phaseNOR, cmap='inferno')
    # plt.colorbar()
    # plt.title("slm_phaseNOR")
    # plt.show()

    print("mk phuz 4 skm")
    # phuzGen.diviX = 10
    # phuzGen.diviY = 10
    phuzGen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": True}
    # phuzGen.linear_grating()
    phuzGen.grat = slm_phaseNOR
    phuzGen._make_full_slm_array()
    phi_centre = phuzGen.final_phuz
    phi_centre = normalize(phi_centre)*220

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
        # # cam_roi_pos = [874, 874]  # grat 10 [1230:1530, 1080:1380]
        # cam_roi_sz = [300, 300]  # grat 10
        # cam_obj.roi_set_roi(int(cam_roi_pos[0] * cam_obj.bin_sz), int(cam_roi_pos[1] * cam_obj.bin_sz),
        #                     int(cam_roi_sz[0] * cam_obj.bin_sz), int(cam_roi_sz[1] * cam_obj.bin_sz))

        cam_obj.stop_acq()
        cam_obj.exposure = 0.1/1000
        cam_obj.prep_acq()
        cam_obj.take_image()
        imgzaz = cam_obj.last_frame
        cam_obj.exposure = exp_time

        plo_che = True
        if plo_che:
            fig = plt.figure()
            # plt.imshow(imgzaz, cmap='inferno', vmax=1500)  # grat 10
            plt.imshow(imgzaz[1156:1456, 1170:1470], cmap='inferno', vmax=500)  # grat 10
            plt.colorbar()
            plt.title("ROi IMG")
            # plt.show()
            plt.show(block=False)
            plt.pause(0.8)
            plt.close(fig)

        print(Fore.LIGHTGREEN_EX + "record background" + Style.RESET_ALL)

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

        # img = np.zeros((300, 300, aperture_number ** 2))
        img = np.zeros((bckgr.shape[0], bckgr.shape[1], aperture_number ** 2))

    # img = np.zeros((roi[1], roi[0], aperture_number ** 2))
    aperture_power = np.zeros(aperture_number ** 2)
    # slm_phase = normalize(slm_phase)*200phi_centre
    slm_phase = phi_centre[:aperture_width, 124:124+aperture_width]

    # figph = plt.figure()
    # plt.subplot(121), plt.imshow(phi_centre, cmap='inferno')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.title("phi_centre")
    # plt.subplot(122), plt.imshow(slm_phase, cmap='inferno')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.title("slm_phase")
    # plt.show()
    # plt.show(block=False)
    # plt.pause(0.8)
    # plt.close(figph)

    plot_within = True
    # here = [108, 109, 110]
    # here = [434, 435, 436]
    dt = np.zeros(aperture_number ** 2)

    slm_phase = np.copy(phi_centre)

    # many_iter = range(aperture_number ** 2)
    # dem_vz = [8, 7, 6, 5, 4, 3, 2, 1]
    dem_vz = [1]
    "# concentric baggage"
    x = np.arange(0, slm_phase.shape[1])
    y = np.arange(0, slm_phase.shape[0])
    cx = (slm_phase.shape[1] // 2) - 6
    cy = (slm_phase.shape[0] // 2) - 7
    arrz = None

    # many_iter = len(dem_vz) + 1
    many_iter = 1
    print('many_iter {}'.format(many_iter))
    for i in range(many_iter):
        print('range(many_iter) {}'.format(range(many_iter)))
        print("iter {} of {}".format(i, aperture_number ** 2))
        t_start = time.time()
        masked_phase = slm_phase

        if not i == 4:  # len(dem_vz) + 0:
            print('is {}'.format(i))
            aha_aa = 0
            a_ha_aa = 0
            for j in range(slm_phase.shape[0]):
                if aha_aa <= aperture_width:
                    # print('jah be cOOl {}'.format(j))
                    aha_aa += 1
                else:
                    if a_ha_aa <= dem_vz[i]*aperture_width:
                        # print('jah be here {}, a_ha_a_a {}'.format(j, a_ha_aa))
                        masked_phase[j, :] = 0
                        # masked_phase[:, j] = 0
                        a_ha_aa += 1
                    else:
                        aha_aa = 0
                        a_ha_aa = 0
                        # print('something wicked be happenin {}, aha_a_a {}, a_ha_a_a {}'.format(j, aha_aa, a_ha_aa))
                        # masked_phase[j, :] = 0
            aha_aa = 0
            a_ha_aa = 0
            for j in range(slm_phase.shape[1]):
                if aha_aa <= aperture_width:
                    # print('jah be cOOl {}'.format(j))
                    aha_aa += 1
                else:
                    if a_ha_aa <= dem_vz[i]*aperture_width:
                        # print('jah be here {}, a_ha_a_a {}'.format(j, a_ha_aa))
                        masked_phase[:, j] = 0
                        a_ha_aa += 1
                    else:
                        aha_aa = 0
                        a_ha_aa = 0
                        # print('something wicked be happenin {}, aha_a_a {}, a_ha_a_a {}'.format(j, aha_aa, a_ha_aa))

        for k in range(0, 13, 2):
            print(k)
            arr = np.zeros((y.size, x.size))
            if k == 0:
                r = 9
                # r = 47  # 47 to erase
            else:
                r = 47*k  # 47 leaves center

            # The two lines below could be merged, but I stored the mask
            # for code clarity.
            mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
            if k == 0:
                arr[mask] = 1  # 1 to erase
            else:
                arr[mask] = 0  # 1 to erase
            if k == 0:
                # r = 9  # 9 to erase
                mask2 = mask
            else:
                r = 9*3*(k + k//1.5)  # 1.5 & 2 are ok-ish
                mask2 = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
            if not k == 0:
                arr[mask2] = 1  # 0 to erase
                mask[mask2] = 0  # 0 to erase

            # trypio = masked_phase  # use as mask to erase from
            trypio = np.zeros((slm_phase.shape[0], slm_phase.shape[1]))
            trypa = trypio
            trypio[mask] = 1  # 0 to erase
            trypio = trypio*slm_phase
            if arrz is None:
                arrz = trypio
            else:
                arrz = arrz + trypio

            # figRIng = plt.figure()
            # plt.subplot(221), plt.imshow(trypio, cmap='inferno')
            # plt.colorbar(fraction=0.046, pad=0.04)
            # plt.title('k: {}'.format(k))
            # plt.subplot(222), plt.imshow(arr, cmap='inferno')
            # plt.colorbar(fraction=0.046, pad=0.04)
            # plt.title('circle is')
            # plt.subplot(223), plt.imshow(slm_phase, cmap='inferno')
            # plt.colorbar(fraction=0.046, pad=0.04)
            # plt.title('slm_phase')
            # plt.subplot(224), plt.imshow(mask, cmap='inferno')
            # plt.colorbar(fraction=0.046, pad=0.04)
            # plt.title('maskito')
            # plt.show()
            # plt.show(block=False)
            # plt.pause(0.8)
            # plt.close(figRIng)

            if not k == 12:
                slm_disp_obj.display(trypio)
            else:
                trypa = arrz
                slm_disp_obj.display(trypa)
            cam_obj.take_average_image(frame_num)
            img[..., i] = cam_obj.last_frame - bckgr

            # aperture_power[i] = np.sum(img[..., i]) / (np.size(img[..., i]) * exp_time)
            aperture_power[i] = 1
            print(aperture_power[i])

            if plot_within:
                fig = plt.figure()
                if not k == 12:
                    plt.subplot(221), plt.imshow(trypio, cmap='inferno')
                else:
                    plt.subplot(221), plt.imshow(trypa, cmap='inferno')
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title('slm_phase')
                plt.subplot(222), plt.imshow(img[..., i][1156:1456, 1170:1470], cmap='inferno', vmin=0, vmax=20)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title('ROI IS')
                plt.subplot(223), plt.imshow(img[..., i], cmap='inferno', vmin=0, vmax=20)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title("full img")
                plt.subplot(224), plt.imshow(arr, cmap='inferno')
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title('circle')
                # plt.show()
                plt.show(block=False)
                fig.savefig(path + '\\iter_{}'.format(k) + '_full.png', dpi=300, bbox_inches='tight',
                            transparent=False)  # True trns worls nice for dispersion thinks I
                plt.pause(0.8)
                plt.close(fig)

                np.save(path + '\\iter_{}'.format(k) + '_full_11', img)
                if not k == 12:
                    np.save(path + '\\iter_{}'.format(k) + '_phuzz', trypio)
                else:
                    np.save(path + '\\iter_{}'.format(k) + '_phuzz', trypa)

        dt[i] = time.time() - t_start
        print("time of iter: {}".format(dt[i]))

        slm_phase = np.copy(phi_centre)

    # cam_obj.stop()
    # cam_obj.roi = roi_mem

    # np.save(path + '//img', img)
    # np.save(path + '//aperture_power', aperture_power)
    #
    # # Find SLM intensity profile
    # i_rec = np.reshape(aperture_power, (aperture_number, aperture_number))
    # # Save data
    # np.save(path + '//i_rec', i_rec)
    #
    # fig = plt.figure()
    # plt.imshow(i_rec, cmap='inferno')
    # plt.colorbar()
    # plt.title("i_rec")
    # plt.show()
    #
    # # Fit Gaussian to measured intensity
    # extent_slm = (slm_disp_obj.slm_size[0] + aperture_width * slm_disp_obj.pitch) / 2
    # x_fit = np.linspace(-extent_slm, extent_slm, aperture_number)
    # x_fit, y_fit = np.meshgrid(x_fit, x_fit)
    # sig_x, sig_y = pms_obj.beam_diameter, pms_obj.beam_diameter
    # popt_slm, perr_slm = ft.fit_gaussian(i_rec, dx=0, dy=0, sig_x=sig_x, sig_y=sig_y, xy=[x_fit, y_fit])
    #
    # i_fit_slm = pt.gaussian(slm_disp_obj.meshgrid_slm[0], slm_disp_obj.meshgrid_slm[1], *popt_slm)
    #
    # # Plotting
    # extent_slm_mm = extent_slm * 1e3
    # extent = [-extent_slm_mm, extent_slm_mm, -extent_slm_mm, extent_slm_mm]
    #
    # fig, axs = plt.subplots(1, 2)
    # divider = make_axes_locatable(axs[0])
    # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    # im = axs[0].imshow(i_rec / np.max(i_rec), cmap='turbo', extent=extent)
    # axs[0].set_title('Intensity at SLM Aperture', fontname='Cambria')
    # axs[0].set_xlabel("x [mm]", fontname='Cambria')
    # axs[0].set_ylabel("y [mm]", fontname='Cambria')
    #
    # divider = make_axes_locatable(axs[1])
    # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    # fig.add_axes(ax_cb)
    # im = axs[1].imshow(i_fit_slm / np.max(i_fit_slm), cmap='turbo', extent=extent)
    # axs[1].set_title('Fitted Gaussian', fontname='Cambria')
    # axs[1].set_xlabel("x [mm]", fontname='Cambria')
    # axs[1].set_ylabel("y [mm]", fontname='Cambria')
    # cbar = plt.colorbar(im, cax=ax_cb)
    # cbar.set_label('normalised intensity', fontname='Cambria')
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(img[..., (aperture_number ** 2 - aperture_number) // 2], cmap='turbo')
    # plt.title('Camera image of central sub-aperture')
    # plt.show()
    #
    # # Save data
    # # np.save(path + '//i_rec', i_rec)
    # np.save(path + '//i_fit_slm', i_fit_slm)
    # np.save(path + '//popt_slm', popt_slm)
    return path + '//i_rec'