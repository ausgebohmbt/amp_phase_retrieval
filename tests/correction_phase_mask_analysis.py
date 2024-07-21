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
from colorama import Fore, Style  # , Back

from experiment import Params
import patterns as pt
import fitting as ft

from slm.helpers import normalize, unimOD, closest_arr, draw_circle, center_overlay
from slm.phase_generator_bOOm import phagen
import cv2


pms_obj = Params()

measure_slm_intensity = False   # Measure the constant intensity at the SLM (laser beam profile)?
measure_slm_phase = False       # Measure the constant phase at the SLM?
correction_section = True
saVe_plo = True
tiltin = True


"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
" create correction phase mask and apply it to correct LG order 2 ~~~~~~~"
"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
path_phase = ("E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\"
                  "amphase_result\\24-07-19_15-47-28_measure_slm_wavefront\\")

if correction_section:
    print('san tsouleriko')
    dphi = np.load(path_phase + "dphi.npy")  # ~~

    zator = normalize(dphi)

    dphi_err = np.load(path_phase + "dphi_err.npy")  # ~~
    a = (dphi_err.shape[0] // 2)
    # b=dphi_err[a:a+1, :]

    there_this = closest_arr(dphi_err[a:a+1, :][0], 620000.)

    # plt.imshow(dphi_err[a:a+1, :], cmap='inferno')
    # plt.colorbar()
    # plt.title("dphi_err idx, {}".format(there_this))
    # plt.show()
    missing_dphi = np.mean( [dphi[a:a+1, there_this[0]-1], dphi[a:a+1, there_this[0]+1]])
    print("l {}, r {}, m {}".format(dphi[a:a+1, there_this[0]-1], dphi[a:a+1, there_this[0]+1], missing_dphi))
    dphi[a:a+1, there_this[0]] = missing_dphi


    missing = np.mean( [zator[a:a+1, there_this[0]-1], zator[a:a+1, there_this[0]+1]])
    print("l {}, r {}, m {}".format(zator[a:a+1, there_this[0]-1], zator[a:a+1, there_this[0]+1], missing))
    zator[a:a+1, there_this[0]] = missing

    # plt.subplot(121)
    # plt.imshow(dphi, cmap='inferno')
    # plt.colorbar()
    # plt.title("dphi")
    # plt.subplot(122)
    # plt.imshow(zator, cmap='inferno')
    # plt.colorbar()
    # plt.title("zator")
    # plt.show()

    inv_zator = - zator
    inv_zator_norm = normalize(inv_zator)
    resu = zator + inv_zator
    resu_norm = zator + inv_zator_norm
    resu_norm_mo = unimOD(resu_norm)
    res_resz = cv2.resize(inv_zator_norm, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)

    # figD = plt.Figure()
    plt.subplot(231)
    plt.imshow(zator, cmap='inferno')
    plt.title("zator")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(232)
    plt.imshow(inv_zator, cmap='inferno')
    plt.title("inv_zator")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(233)
    plt.imshow(inv_zator_norm, cmap='inferno')
    plt.title("inv_zator_norm")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(234)
    plt.imshow(resu, cmap='inferno')
    plt.title("resu")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(235)
    plt.imshow(resu_norm, cmap='inferno')
    plt.title("resu_norm")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(236)
    plt.imshow(dphi, cmap='inferno')
    plt.title("dphi")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()

    plt.subplot(121)
    plt.imshow(res_resz, cmap='inferno')
    plt.title("res_resz")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(122)
    plt.imshow(inv_zator_norm, cmap='inferno')
    plt.title("inv_zator_norm")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()

    "cre'te spiral phase and start testin"
    phagen.modDepth = 255
    phagen.crxn_phuz_boom = res_resz
    phagen.state_lg  = True
    phagen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": True,
                           "lg_pol": True, "corr_phase": False}
    phagen.linear_grating()
    phagen._make_full_slm_array()
    slm_phase_init = phagen.final_phuz
    # plt.imshow(slm_phase, cmap='inferno')
    # plt.title("slm_phase")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.show()

    radios = 500
    circular_aperture = draw_circle(1024, radios)
    circular_aperture = center_overlay(1272, 1024, circular_aperture)

    slm_phase = slm_phase_init * circular_aperture


"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

if tiltin:
    print('san tsouleriko, o o o o o')
    dphi = np.load(path_phase + "dphi.npy")  # ~~
    i_fit_mask = np.load(path_phase + "i_fit_mask.npy")  # ~~
    roi_min_x = 0
    roi_min_y = 0
    roi_n = 30
    aperture_number = 30
    # Determine phase
    dphi_uw_nopad = pt.unwrap_2d(dphi)
    dphi_uw_notilt = ft.remove_tilt(dphi_uw_nopad)
    pad_roi = ((roi_min_x, aperture_number - roi_n - roi_min_x), (roi_min_y, aperture_number - roi_n - roi_min_y))
    dphi_uw = np.pad(dphi_uw_nopad, pad_roi)
    dphi_uw_mask = pt.unwrap_2d_mask(dphi, i_fit_mask)
    dphi_uw_mask = np.pad(dphi_uw_mask, pad_roi)

    # this be it
    zator = normalize(dphi_uw_mask)
    # zator = normalize(dphi)


    # # # # figD = plt.Figure()
    # plt.subplot(231)
    # plt.imshow(dphi, cmap='inferno')
    # plt.title("dphi")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.subplot(232)
    # plt.imshow(dphi_uw_nopad, cmap='inferno')
    # plt.title("dphi_uw_nopad")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.subplot(233)
    # plt.imshow(dphi_uw_notilt, cmap='inferno')
    # plt.title("dphi_uw_notilt")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.subplot(234)
    # plt.imshow(dphi_uw, cmap='inferno')
    # plt.title("dphi_uw")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.subplot(235)
    # plt.imshow(dphi_uw_mask, cmap='inferno')
    # plt.title("dphi_uw_mask")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.subplot(236)
    # # plt.imshow(resu_norm_mo, cmap='inferno')
    # # plt.title("resu_norm_mo")
    # # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.show()



    dphi_err = np.load(path_phase + "dphi_err.npy")  # ~~
    i_fit_mask = np.load(path_phase + "i_fit_mask.npy")  # ~~
    roi_min_x = 0
    roi_min_y = 0
    roi_n = 30
    aperture_number = 30
    # Determine phase
    dphi_uw_nopad = pt.unwrap_2d(dphi)
    dphi_uw_notilt = ft.remove_tilt(dphi_uw_nopad)
    pad_roi = ((roi_min_x, aperture_number - roi_n - roi_min_x), (roi_min_y, aperture_number - roi_n - roi_min_y))
    dphi_uw = np.pad(dphi_uw_nopad, pad_roi)
    dphi_uw_mask = pt.unwrap_2d_mask(dphi, i_fit_mask)
    dphi_uw_mask = np.pad(dphi_uw_mask, pad_roi)
    a = (dphi_err.shape[0] // 2)
    # b=dphi_err[a:a+1, :]

    there_this = closest_arr(dphi_err[a:a+1, :][0], 620000.)

    # plt.imshow(dphi_err[a:a+1, :], cmap='inferno')
    # plt.colorbar()
    # plt.title("dphi_err idx, {}".format(there_this))
    # plt.show()

    missing_dphi_uw_mask = np.mean( [dphi_uw_mask[a:a+1, there_this[0]-1], dphi_uw_mask[a:a+1, there_this[0]+1]])
    print("l {}, r {}, m {}".format(dphi_uw_mask[a:a+1, there_this[0]-1], dphi_uw_mask[a:a+1, there_this[0]+1],
                                    missing_dphi_uw_mask))
    dphi_uw_mask[a:a+1, there_this[0]] = missing_dphi_uw_mask

    missing = np.mean( [zator[a:a+1, there_this[0]-1], zator[a:a+1, there_this[0]+1]])
    print("l {}, r {}, m {}".format(zator[a:a+1, there_this[0]-1], zator[a:a+1, there_this[0]+1], missing))
    zator[a:a+1, there_this[0]] = missing

    # plt.subplot(121)
    # plt.imshow(dphi, cmap='inferno')
    # plt.colorbar()
    # plt.title("dphi")
    # plt.subplot(122)
    # plt.imshow(zator, cmap='inferno')
    # plt.colorbar()
    # plt.title("zator")
    # plt.show()

    inv_zator = - zator
    inv_zator_norm = normalize(inv_zator)
    resu = zator + inv_zator
    resu_norm = zator + inv_zator_norm
    resu_norm_mo = unimOD(resu_norm)
    res_resz = cv2.resize(inv_zator_norm, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)

    # figD = plt.Figure()
    plt.subplot(231)
    plt.imshow(zator, cmap='inferno')
    plt.title("zator")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(232)
    plt.imshow(inv_zator, cmap='inferno')
    plt.title("inv_zator")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(233)
    plt.imshow(inv_zator_norm, cmap='inferno')
    plt.title("inv_zator_norm")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(234)
    plt.imshow(resu, cmap='inferno')
    plt.title("resu")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(235)
    plt.imshow(resu_norm, cmap='inferno')
    plt.title("resu_norm")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(236)
    plt.imshow(dphi_uw_mask, cmap='inferno')
    plt.title("dphi_uw_mask")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()

    plt.subplot(121)
    plt.imshow(res_resz, cmap='inferno')
    plt.title("res_resz")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(122)
    plt.imshow(inv_zator_norm, cmap='inferno')
    plt.title("inv_zator_norm")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()

    "cre'te spiral phase and start testin"
    phagen.modDepth = 255
    phagen.crxn_phuz_boom = res_resz
    phagen.state_lg  = True
    phagen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": True,
                           "lg_pol": True, "corr_phase": False}
    phagen.linear_grating()
    phagen._make_full_slm_array()
    slm_phase_init = phagen.final_phuz
    # plt.imshow(slm_phase, cmap='inferno')
    # plt.title("slm_phase")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.show()


    radios = 500
    circular_aperture = draw_circle(1024, radios)
    circular_aperture = center_overlay(1272, 1024, circular_aperture)

    slm_phase = slm_phase_init * circular_aperture

    "enable corection"
    radios = 500
    circular_aperture = draw_circle(1024, radios)
    circular_aperture = center_overlay(1272, 1024, circular_aperture)

    "upload 2 slm"
    aperture_init_phase = slm_phase_init * circular_aperture

    # correct
    phagen.state_cor_phase = True
    phagen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": False,
                           "lg_pol": True, "corr_phase": True}
    phagen._make_full_slm_array()
    slm_phase = phagen.final_phuz

    slm_phase = phagen.final_phuz * circular_aperture

print(Fore.LIGHTBLUE_EX + "jah ha run" + Style.RESET_ALL)
print('es el finAl')
# 'es el finAl'
