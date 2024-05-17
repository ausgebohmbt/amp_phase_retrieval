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
# from matplotlib import cm
# from colorama import Fore, Style  # , Back

import measurement_functions as mfunc
import error_metrics as m, patterns as pt, fitting as ft
from mpl_toolkits.axes_grid1 import make_axes_locatable
from slm.phase_generator import phagen as phuzGen
from slm.helpers import unimOD, center_overlay  # , center_crop, tiler, normalize
from experiment import Params  # , Camera, SlmDisp


pms_obj = Params()
# print("pms_obj.k {}".format(pms_obj.k))
# print("lamda {} & lamda under 2pi {}".format(pms_obj.wavelength, 2*np.pi / pms_obj.wavelength))

slm_disp_obj = None
cam_obj = None
exp = 100
cam_roi_sz = [300, 300]
fl = pms_obj.fl
fit_sine = ft.FitSine(fl, pms_obj.k)

plots_o_intensity = False
plots_o_phase = True
saVe_plo = False

path_intense = ("E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\"
                "amphase_result\\24-05-15_14-22-49_measure_slm_intensity\\")

path_phase = ("E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\"
              "amphase_result\\24-05-15_14-41-15_measure_slm_wavefront\\")

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
" Intensity Plots ~~~~~ Intensity Plots ~~~~~ Intensity Plots ~~~~~ Intense ~"
"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
if plots_o_intensity:
    i_rec = np.load(path_intense + "i_rec.npy")
    i_fit_slm = np.load(path_intense + "i_fit_slm.npy")
    img = np.load(path_intense + "img.npy")
    popt_slm = np.load(path_intense + "popt_slm.npy")
    aperture_power = np.load(path_intense + "aperture_power.npy")

    loPhuz = plt.figure()
    plt.subplot(121), plt.imshow(i_rec, cmap='turbo')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('i_rec')
    plt.subplot(122), plt.imshow(i_fit_slm, cmap='turbo')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('i_rec')
    if saVe_plo:
        plt.show(block=False)
        # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
        loPhuz.savefig(path_intense[:-9] + '\\int.png', dpi=300, bbox_inches='tight',
                       transparent=False)  # True trns worls nice for dispersion thinks I
        plt.pause(0.8)
        plt.close()
    else:
        plt.show(block=False)
        plt.pause(0.8)
        plt.close()

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
" Phase Plots ~~~~~ Phase Plots ~~~~~ Phase Plots ~~~~~ Phase Wha?? ~~~~~"
"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
if plots_o_phase:

    try:
        img = np.load(path_phase + "img.npy")
        imgzz = np.load(path_phase + "imgzz.npy")
    except Exception as e:
        print('no img sh__, fu__, stacks')
    powah = np.load(path_phase + "powah.npy")
    power = np.load(path_phase + "power.npy")
    i_fit = np.load(path_phase + "i_fit.npy")
    i_fit_mask = np.load(path_phase + "i_fit_mask.npy")
    popt_sv = np.load(path_phase + "popt_sv.npy")
    perr_sv = np.load(path_phase + "perr_sv.npy")
    dphi_uw_mask = np.load(path_phase + "dphi_uw_mask.npy")
    dphi_uw = np.load(path_phase + "dphi_uw.npy")
    dphi_err = np.load(path_phase + "dphi_err.npy")
    dphi = np.load(path_phase + "dphi.npy")
    dx = np.load(path_phase + "dx.npy")
    dy = np.load(path_phase + "dy.npy")
    tt = np.load(path_phase + "t.npy")

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
        fig1.savefig(path_phase + 'fit.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.8)
        plt.close(fig1)
    else:
        # plt.show()
        plt.show(block=False)
        plt.pause(0.8)
        plt.close()

    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    " ~~ dphi_n_err ~~~ dphi_n_err ~~~~~~ dphi_n_err ~~~~~~"
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    fig = plt.figure()
    plt.subplot(121), plt.imshow(dphi, cmap='inferno')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('dphi')
    plt.subplot(122), plt.imshow(dphi_err, cmap='inferno')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('dphi_err')
    if saVe_plo:
        plt.show(block=False)
        fig.savefig(path_phase + 'dphi_n_err.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.8)
        plt.close(fig)
    else:
        # plt.show()
        plt.show(block=False)
        plt.pause(0.8)
        plt.close()

    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

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
        figPow.savefig(path_phase +'\\powaher.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.8)
        plt.close(figPow)
    else:
        # plt.show()
        plt.show(block=False)
        plt.pause(0.8)
        plt.close()

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
    #     figD.savefig(path_phase +'\\detPhuz.png', dpi=300, bbox_inches='tight', transparent=False)
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
        figD.savefig(path_phase +'\\detPhuz.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.8)
        plt.close(figD)
    else:
        # plt.show()
        plt.show(block=False)
        plt.pause(0.8)
        plt.close()

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
" hOlO testz ~~~~~ hOlO testz ~~~~~ hOlO testz ~~~~~ hOlOw checkz ~~~~~~"
"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
test_phuz = False
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
    # plt.show()
    plt.show(block=False)
    plt.pause(0.8)
    plt.close()
    # if saVe_plo:
    #     plt.show(block=False)
    #     # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
    #     figD.savefig(path_phase +'\\detPhuz.png', dpi=300, bbox_inches='tight', transparent=False)
    #     plt.pause(2)
    #     plt.close(figD)
    # else:
    #     plt.show()

print('es el finAl')
# 'es el finAl'
