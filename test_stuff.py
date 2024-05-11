"""
Feedback algorithm example
==========================

This script calculates phase patterns for a phase-modulating liquid crystal on silicon (LCOS) spatial light modulator
(SLM) to create accurate light potentials by modelling pixel crosstalk on the SLM and using conjugate gradient (CG)
minimisation with camera feedback (see https://doi.org/10.1038/s41598-023-30296-6).

Using this script, it should be easy to switch between the different patterns from our publication, turn on pixel
crosstalk modelling and switch between the fast Fourier transform (FFT) and the angular spectrum method (ASM) to model
the propagation of light.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# import calibrate_slm as clb
import measurement_functions as mfunc
import error_metrics as m, patterns as pt, fitting as ft
from mpl_toolkits.axes_grid1 import make_axes_locatable
from slm.phase_generator import phagen as phuzGen
from slm.helpers import unimOD, normalize, center_overlay, center_crop, tiler

from experiment import Params#, Camera, SlmDisp

from colorama import Fore, Style  # , Back

pms_obj = Params()
print("pms_obj.k")
print(pms_obj.k)
print("lamda")
print(pms_obj.wavelength)
print(2 * np.pi / pms_obj.wavelength)

slm_disp_obj = None
cam_obj = None
exp = 100
cam_roi_sz = [300, 300]
fl = pms_obj.fl
fit_sine = ft.FitSine(fl, pms_obj.k)

measure_slm_intensity = False   # Measure the constant intensity at the SLM (laser beam profile)?
measure_slm_phase = False       # Measure the constant phase at the SLM?

"Measuring the constant intensity and phase at the SLM"
if measure_slm_intensity is True:

    i_path = mfunc.measure_slm_intensity(slm_disp_obj, cam_obj, pms_obj,
                                       15, 64, exp/1000,
                                       256, np.asarray(cam_roi_sz[0]))
    pms_obj.i_path = i_path
if measure_slm_phase is True:
    pass
    # phi_path = clb.measure_slm_wavefront(slm_disp_obj, cam_obj, pms_obj, 30, 16, 64, 40000, 256, roi_min_x=2,
    #                                      roi_min_y=2, roi_n=26)
    phi_path = mfunc.measure_slm_wavefront(slm_disp_obj, cam_obj, pms_obj, 30, 16,
                                         64, 40000, 256, n_avg_frames=5, roi_min_x=0,
                                         roi_min_y=0, roi_n=30)
    # pms_obj.phi_path = phi_path


load_existing = False
saVe_plo = True
# this_path = pms_obj.phi_path
# this_path = pms_obj.i_path
this_path = "E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\amphase_result\\24-05-10_10-55-51_measure_slm_intensity\\i_rec.npy"
# this_path = "E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\amphase_result\\24-05-10_12-15-14_measure_slm_wavefront\\power.npy"

if load_existing:
    loaded_phuz = np.load(this_path)

    loPhuz = plt.figure()
    plt.imshow(loaded_phuz, cmap='turbo')
    # plt.imshow(loaded_phuz / np.pi / 2, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    # plt.title('intense')
    # plt.title('Unwrapped measured phase')
    plt.show()

    if saVe_plo:
        plt.show(block=False)
        # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
        loPhuz.savefig(this_path[:-9] +'\\int.png', dpi=300, bbox_inches='tight',
                    transparent=False)  # True trns worls nice for dispersion thinks I
        plt.pause(2.4)
        plt.close()
    else:
        plt.show()

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
data_plots = False
if data_plots:
    # main_path = "E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\amphase_result\\24-05-10_22-26-07_measure_slm_wavefront\\"
    main_path = "E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\amphase_result\\24-05-11_00-13-42_measure_slm_wavefront\\"

    img = np.load(main_path + "img.npy")
    popt_sv = np.load(main_path + "popt_sv.npy")
    powah = np.load(main_path + "powah.npy")
    power = np.load(main_path + "power.npy")
    i_fit = np.load(main_path + "i_fit.npy")

    # plt.imshow(img[:, :, -1], cmap='turbo')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.title("img[:, :, -1]")
    # plt.show()

    x, y = pt.make_grid(img[:, :, 0], scale=12.5e-6)
    x_data = np.vstack((x.ravel(), y.ravel()))
    img_size = img.shape[0]
    # fit_test = np.reshape(fit_sine.fit_sine(x_data, *popt_sv[-1]), (img_size, img_size))

    fig1 = plt.Figure()
    plt.subplot(121)
    plt.imshow(img[:, :, -1], cmap='turbo')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("img[:, :, -1]")
    plt.subplot(122)
    plt.imshow(i_fit, cmap='turbo')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("i_fit")
    if saVe_plo:
        plt.show(block=False)
        # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
        fig1.savefig(main_path + 'fit.png', dpi=600, bbox_inches='tight', transparent=False)
        plt.pause(0.4)
        plt.close(fig1)
    else:
        plt.show()

    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    # figPow, ax = plt.subplots(1, 2)
    # # plt.subplot(121)
    # ax[0].imshow(powah, cmap='turbo')
    # ax[0].set_title("powah")
    # figPow.colorbar(cm.ScalarMappable(cmap="turbo"), ax=ax[0], fraction=0.046, pad=0.04)
    # # figPow.colorbar(mappable='turbo', ax=ax[0], shrink=0.6)
    # # ax[0].colorbar(fraction=0.046, pad=0.04)
    # # plt.subplot(122)
    # ax[1].imshow(power, cmap='turbo')
    # ax[1].set_title("power")
    # figPow.colorbar(cm.ScalarMappable(norm=None, cmap="turbo"), ax=ax[1], fraction=0.046, pad=0.04)
    # # ax[1].colorbar(fraction=0.046, pad=0.04)
    # if saVe_plo:
    #     plt.show(block=False)
    #     # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
    #     figPow.savefig(main_path +'\\powaher.png', dpi=300, bbox_inches='tight', transparent=False)
    #     plt.pause(0.4)
    #     plt.close(figPow)
    # else:
    #     plt.show()

    figPow, axs = plt.subplots(1, 2)
    divider = make_axes_locatable(axs[0])
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    im = axs[0].imshow(powah, cmap='turbo')
    # im = axs[0].imshow(i_rec / np.max(i_rec), cmap='turbo', extent=extent)
    axs[0].set_title('powah', fontname='Cambria')
    # axs[0].set_xlabel("x [mm]", fontname='Cambria')
    # axs[0].set_ylabel("y [mm]", fontname='Cambria')

    divider = make_axes_locatable(axs[1])
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    figPow.add_axes(ax_cb)
    im = axs[1].imshow(power, cmap='turbo')
    # im = axs[1].imshow(i_fit_slm / np.max(i_fit_slm), cmap='turbo', extent=extent)
    axs[1].set_title('power', fontname='Cambria')
    # axs[1].set_xlabel("x [mm]", fontname='Cambria')
    # axs[1].set_ylabel("y [mm]", fontname='Cambria')
    cbar = plt.colorbar(im, cax=ax_cb)
    cbar.set_label('intensity FIX ME', fontname='Cambria')
    # plt.show()
    if saVe_plo:
        plt.show(block=False)
        # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
        figPow.savefig(main_path +'\\powaher.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.4)
        plt.close(figPow)
    else:
        plt.show()





    dphi = np.load(main_path + "dphi.npy")

    # Determine phase
    dphi_uw_nopad = pt.unwrap_2d(dphi)
    dphi_uw_notilt = ft.remove_tilt(dphi_uw_nopad)

    # figD = plt.Figure()
    # plt.subplot(121)
    # plt.imshow(dphi_uw_nopad, cmap='turbo')
    # plt.title("dphi_uw_nopad")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.subplot(122)
    # plt.imshow(dphi_uw_notilt, cmap='turbo')
    # plt.title("dphi_uw_notilt")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # if saVe_plo:
    #     plt.show(block=False)
    #     # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
    #     figD.savefig(main_path +'\\detPhuz.png', dpi=300, bbox_inches='tight', transparent=False)
    #     plt.pause(2)
    #     plt.close(figD)
    # else:
    #     plt.show()

    figD, axs = plt.subplots(1, 2)
    divider = make_axes_locatable(axs[0])
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    im = axs[0].imshow(dphi_uw_nopad, cmap='turbo')
    # im = axs[0].imshow(i_rec / np.max(i_rec), cmap='turbo', extent=extent)
    axs[0].set_title('dphi_uw_nopad1', fontname='Cambria')
    # axs[0].set_xlabel("x [mm]", fontname='Cambria')
    # axs[0].set_ylabel("y [mm]", fontname='Cambria')

    divider = make_axes_locatable(axs[1])
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    figD.add_axes(ax_cb)
    im = axs[1].imshow(dphi_uw_notilt, cmap='turbo')
    # im = axs[1].imshow(i_fit_slm / np.max(i_fit_slm), cmap='turbo', extent=extent)
    axs[1].set_title('dphi_uw_notilt', fontname='Cambria')
    # axs[1].set_xlabel("x [mm]", fontname='Cambria')
    # axs[1].set_ylabel("y [mm]", fontname='Cambria')
    cbar = plt.colorbar(im, cax=ax_cb)
    cbar.set_label('intensity FIX ME', fontname='Cambria')
    # plt.show()
    if saVe_plo:
        plt.show(block=False)
        # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
        figD.savefig(main_path +'\\detPhuz.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.4)
        plt.close(figD)
    else:
        plt.show()


test_phuz = True
if test_phuz:

    npix = 1024
    res_y, res_x = 1024, 1272
    border_x = int(np.abs(res_x - res_y) // 2)

    spot_pos = 256
    aperture_number = 30
    aperture_width = 32

    zeros = np.zeros((npix, npix))
    zeros_full = np.zeros((npix, 1272))

    lin_phase = np.array([-spot_pos, -spot_pos])
    slm_phase4me = mfunc.init_pha(np.zeros((1024, 1272)), 1272, 12.5e-6,
                         pms_obj, lin_phase=lin_phase)
    # slm_phaseOUT = pt.init_phase(np.zeros((aperture_width, aperture_width)), slm_disp_obj,
    #                      pms_obj, lin_phase=lin_phase)
    linFlip = np.fliplr(slm_phase4me)

    # slm_phase = pt.init_phase(np.zeros((aperture_width, aperture_width)), slm_disp_obj, pms_obj, lin_phase=lin_phase)
    # slm_phase = np.remainder(slm_phaseOUT, 2 * np.pi)
    slm_phase4me = np.remainder(slm_phase4me, 2 * np.pi)
    slm_phase4me = np.fliplr(slm_phase4me)


    # slm_phase = np.remainder(slm_phase, 2 * np.pi)
    slm_phase = slm_phase4me
    # slm_phase = np.flipud(np.fliplr(slm_phase))
    slm_idx = mfunc.get_aperture_indices(aperture_number, aperture_number, border_x, npix + border_x, 0, npix, aperture_width,
                                   aperture_width)

    # Display central sub-aperture on SLM and check if camera is over-exposed.
    i = (aperture_number ** 2) // 2 - aperture_number // 2
    phi_centre = np.zeros_like(zeros)
    # phi_centre[slm_idx[0][i]:slm_idx[1][i], slm_idx[2][i]:slm_idx[3][i]] = slm_phase
    phi_centre = slm_phase
    slm_phaseNOR = mfunc.normalize(slm_phase)

    # plt.imshow(slm_phaseNOR, cmap='inferno')
    # plt.colorbar()
    # plt.title("slm_phaseNOR")
    # plt.show()

    phuzGen.diviX = 10
    phuzGen.diviY = 10
    phuzGen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": True}
    # phuzGen.linear_grating()
    phuzGen.grat = slm_phaseNOR
    phuzGen._make_full_slm_array()
    phi_centre = phuzGen.final_phuz
    phi_centre = mfunc.normalize(phi_centre)*220

    lg = center_overlay(res_x, res_y, phuzGen.lgPhu_mod)
    lgra = np.add(phuzGen.grat, lg)
    phuz = unimOD(lgra) * phuzGen.modDepth
    lg_gra = phuz.astype('uint8')

    roi_min_x = 0
    roi_min_y = 0
    roi_n = 30
    # Determine phase
    dphi_uw_nopa = pt.unwrap_2d(lg_gra)
    dphi_uw_notil = ft.remove_tilt(dphi_uw_nopa)
    pad_ro = ((roi_min_x, aperture_number - roi_n - roi_min_x), (roi_min_y, aperture_number - roi_n - roi_min_y))
    dph_uw = np.pad(dphi_uw_nopa, pad_ro)
    # dph_uw = mfunc.normalize(dph_uw)
    mumnwra = dph_uw / np.pi / 2


    figD = plt.Figure()
    plt.subplot(231)
    plt.imshow(phuzGen.lgPhu_mod, cmap='inferno')
    plt.title("lg")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(232)
    plt.imshow(dphi_uw_notil, cmap='inferno')
    plt.title("rm piston tilt")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(233)
    plt.imshow(phuzGen.grat, cmap='inferno')
    plt.title("grat")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(234)
    plt.imshow(lg_gra, cmap='inferno')
    plt.title("lg_gra")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(235)
    plt.imshow(linFlip, cmap='inferno')
    plt.title("linFlip")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(236)
    plt.imshow(mumnwra, cmap='inferno')
    plt.title("mumnwra")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()
    # if saVe_plo:
    #     plt.show(block=False)
    #     # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
    #     figD.savefig(main_path +'\\detPhuz.png', dpi=300, bbox_inches='tight', transparent=False)
    #     plt.pause(2)
    #     plt.close(figD)
    # else:
    #     plt.show()


print('es el finAl')
# 'es el finAl'
