"""
Script to  visualise and analyze the results of the Feedback Algorithm Example
==========================
"""

import numpy as np
import matplotlib.pyplot as plt
import os
# from matplotlib import cm
# from colorama import Fore, Style  # , Back
import cv2

# import measurement_functions as mfunc
import error_metrics as m, patterns as pt, fitting as ft
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from slm.phase_generator import phagen as phuzGen
from slm.helpers import normalize, unimOD, closest_arr, draw_circle, center_overlay  #   # , center_crop, tiler,
from experiment import Params  # , Camera, SlmDisp

pms_obj = Params()
# print("pms_obj.k {}".format(pms_obj.k))
# print("lamda {} & lamda under 2pi {}".format(pms_obj.wavelength, 2*np.pi / pms_obj.wavelength))

slm_disp_obj = None
cam_obj = None
exp = 100
cam_roi_sz = [300, 300]
fl = pms_obj.fl
aperture_width_intense = 32
slm_pitch = 12.5e-6
slm_res = np.asarray([1272, 1024], dtype=float)
slm_size = slm_res * slm_pitch   # x, y dimensions of the SLM [m]

fit_sine = ft.FitSine(fl, pms_obj.k)

plots_o_intensity = False
plots_o_phase = False
saVe_plo = False

path_intense = ("E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\"
                "amphase_result\\24-06-07_11-46-50_measure_slm_intensity\\")

path_phase = ("E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\"
              "amphase_result\\24-06-07_12-08-15_measure_slm_wavefront\\")

# # LG data
# path_intense = ("E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\"
#                 "amphase_result\\24-05-10_22-04-36_measure_slm_intensity\\")
# path_phase = ("E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\"
#               "amphase_result\\24-05-11_00-13-42_measure_slm_wavefront\\")

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
" Intensity Plots ~~~~~ Intensity Plots ~~~~~ Intensity Plots ~~~~~ Intense ~"
"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
if plots_o_intensity:
    i_rec = np.load(path_intense + "i_rec.npy")
    i_fit_slm = np.load(path_intense + "i_fit_slm.npy")
    try:
        img = np.load(path_intense + "img.npy")
    except Exception as e:
        print("no img in int folder")
    popt_slm = np.load(path_intense + "popt_slm.npy")
    aperture_power = np.load(path_intense + "aperture_power.npy")

    # Plotting Prep
    extent_slm = (slm_size[0] + aperture_width_intense * slm_pitch) / 2
    extent_slm_mm = extent_slm * 1e3
    extent = [-extent_slm_mm, extent_slm_mm, -extent_slm_mm, extent_slm_mm]

    " Intensity Plot & fit ~"
    fINTg = plt.figure()
    plt.subplot(121), plt.imshow(i_rec, cmap='turbo')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('i_rec')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.subplot(122), plt.imshow(i_fit_slm, cmap='turbo')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('i_fit_slm')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.tight_layout()  # plt.tight_layout(pad=5.0)
    if saVe_plo:
        path = path_intense + "\\analysis"
        if not os.path.exists(path):
            os.mkdir(path)
        plt.show(block=False)
        fINTg.savefig(path + '\\int.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.8)
        plt.close(fINTg)
    else:
        # plt.show()
        plt.show(block=False)
        plt.pause(0.8)
        plt.close(fINTg)

    " Intensity Plot & fit ~~~~ NormaLised"
    fINTgNorm = plt.figure()
    plt.subplot(121), plt.imshow(i_rec / np.max(i_rec), cmap='turbo', extent=extent)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Intensity at SLM Aperture')
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.subplot(122), plt.imshow(i_fit_slm, cmap='turbo', extent=extent)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Fitted Gaussian')
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.tight_layout()  # plt.tight_layout(pad=5.0)
    if saVe_plo:
        plt.show(block=False)
        path = path_intense + "\\analysis"
        if not os.path.exists(path):
            os.mkdir(path)
        fINTgNorm.savefig(path + '\\intNorm.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.8)
        plt.close(fINTgNorm)
    else:
        # plt.show()
        plt.show(block=False)
        plt.pause(0.8)
        plt.close(fINTgNorm)

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
" Phase Plots ~~~~~ Phase Plots ~~~~~ Phase Plots ~~~~~ Phase Wha?? ~~~~~"
"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
if plots_o_phase:

    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    " ~~ loAD npy arrays ~~ "
    try:
        img = np.load(path_phase + "img.npy")  # ~~
        imgzz = np.load(path_phase + "imgzz.npy")
        exist_img = True
    except Exception as e:
        print('no img sh__, fu__, stacks')
        exist_img = False
    try:
        aperture_coverage = np.load(path_phase + "aperture_coverage.npy")  # ~~
        exist_coVer = True
    except Exception as e:
        print('no aperture_coverage saved')
        exist_coVer = False
    powah = np.load(path_phase + "powah.npy")  # ~~
    power = np.load(path_phase + "power.npy")  # ~~
    i_fit = np.load(path_phase + "i_fit.npy")  # ~~
    i_fit_mask = np.load(path_phase + "i_fit_mask.npy")  # ~~
    popt_sv = np.load(path_phase + "popt_sv.npy")  # used 2 mk dphi & fit_test
    # perr_sv = np.load(path_phase + "perr_sv.npy")  # used to mk dphi_err
    dphi_uw_mask = np.load(path_phase + "dphi_uw_mask.npy")
    dphi_uw = np.load(path_phase + "dphi_uw.npy")  # ~~
    dphi_err = np.load(path_phase + "dphi_err.npy")  # ~~
    dphi = np.load(path_phase + "dphi.npy")  # ~~
    dx = np.load(path_phase + "dx.npy")
    dy = np.load(path_phase + "dy.npy")
    tt = np.load(path_phase + "t.npy")

    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    "prep/recre'te stuff"
    if exist_img:
        x, y = pt.make_grid(img[:, :, 0], scale=12.5e-6)
        x_data = np.vstack((x.ravel(), y.ravel()))
        img_size = img.shape[0]
        fit_sine.set_dx_dy(dx, dy)
        fit_test = np.reshape(fit_sine.fit_sine(x_data, *popt_sv[-1]), (img_size, img_size))

        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        " ~~ last image & fit_test ~"
        fgUnPhi = plt.figure()
        plt.subplot(121), plt.imshow(img[:, :, -1], cmap='magma')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('last image')
        plt.xlabel("x pixels")
        plt.ylabel("y pixels")
        plt.subplot(122), plt.imshow(fit_test, cmap='magma')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('fit_test')
        plt.xlabel("x pixels")
        plt.ylabel("y pixels")
        plt.tight_layout()  # plt.tight_layout(pad=5.0)
        if saVe_plo:
            path = path_phase + "\\analysis"
            if not os.path.exists(path):
                os.mkdir(path)
            plt.show(block=False)
            fgUnPhi.savefig(path + '\\last_img_n_fit_o_test_o.png', dpi=300, bbox_inches='tight', transparent=False)
            plt.pause(0.8)
            plt.close(fgUnPhi)
        else:
            plt.show()
            # plt.show(block=False)
            plt.pause(0.8)
            plt.close(fgUnPhi)

    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    " Unwrapped phase & errah ~"
    fgUnPhi = plt.figure()
    plt.subplot(131), plt.imshow(dphi_uw / np.pi / 2, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Unwrapped measured phase')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.subplot(132), plt.imshow(dphi_uw, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Unwrapped measured phase, non_normalised')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.subplot(133), plt.imshow(dphi_err, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('dphi_err')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.tight_layout()  # plt.tight_layout(pad=5.0)
    if saVe_plo:
        path = path_phase + "\\analysis"
        if not os.path.exists(path):
            os.mkdir(path)
        plt.show(block=False)
        fgUnPhi.savefig(path + '\\Unwrapped_dphi_and_err.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.8)
        plt.close(fgUnPhi)
    else:
        plt.show()
        # plt.show(block=False)
        plt.pause(0.8)
        plt.close(fgUnPhi)

    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    " ~~ powah ~~ "
    fgPow = plt.figure()
    plt.subplot(121), plt.imshow(powah, cmap='turbo')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('power of central pixel')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.subplot(122), plt.imshow(power, cmap='turbo')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('power around central pixel [3x3]')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.tight_layout()  # plt.tight_layout(pad=5.0)
    if saVe_plo:
        path = path_phase + "\\analysis"
        if not os.path.exists(path):
            os.mkdir(path)
        plt.show(block=False)
        fgPow.savefig(path + '\\power.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.8)
        plt.close(fgPow)
    else:
        plt.show()
        # plt.show(block=False)
        plt.pause(0.8)
        plt.close(fgPow)

    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    " ~~ mask essentials ~~ "
    fgMasko = plt.figure()
    plt.subplot(121), plt.imshow(dphi_uw_mask, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('dphi_uw_mask')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.subplot(122), plt.imshow(i_fit_mask, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('i_fit_mask')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.tight_layout()  # plt.tight_layout(pad=5.0)
    if saVe_plo:
        path = path_phase + "\\analysis"
        if not os.path.exists(path):
            os.mkdir(path)
        plt.show(block=False)
        fgMasko.savefig(path + '\\phiMask_n_fit.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.8)
        plt.close(fgMasko)
    else:
        plt.show()
        # plt.show(block=False)
        plt.pause(0.8)
        plt.close(fgMasko)

    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    " ~~ mask all ~~ "
    fgMaskAll = plt.figure()
    plt.subplot(221), plt.imshow(dphi_uw_mask, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('dphi_uw_mask')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.subplot(222), plt.imshow(i_fit_mask, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('i_fit_mask')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.subplot(223), plt.imshow(dphi, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('dphi [when unwrapped with i_fit gives _uw_mask]')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.subplot(224), plt.imshow(i_fit, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('i_fit')
    plt.xlabel("x pixels")
    plt.ylabel("y pixels")
    plt.tight_layout()  # plt.tight_layout(pad=5.0)
    if saVe_plo:
        path = path_phase + "\\analysis"
        if not os.path.exists(path):
            os.mkdir(path)
        plt.show(block=False)
        fgMaskAll.savefig(path + '\\fgMaskAll.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.pause(0.8)
        plt.close(fgMaskAll)
    else:
        plt.show()
        # plt.show(block=False)
        plt.pause(0.8)
        plt.close(fgMaskAll)

    " ~~~ coVeRaGe ~~~~~~"
    if exist_coVer:
        figCoV = plt.figure()
        plt.imshow(aperture_coverage)
        plt.title('Coverage of sub-apertures on the SLM')
        plt.xlabel("x pixels")
        plt.ylabel("y pixels")
        if saVe_plo:
            plt.show(block=False)
            path = path_phase + "\\analysis"
            if not os.path.exists(path):
                os.mkdir(path)
            figCoV.savefig(path + '\\coVerAge.png', dpi=300, bbox_inches='tight', transparent=False)
            plt.pause(0.8)
            plt.close(figCoV)
        else:
            plt.show()
            # plt.show(block=False)
            plt.pause(0.8)
            plt.close(figCoV)
    else:
        print("no coverage data saved")


"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
" hOlO testz ~~~~~ hOlO testz ~~~~~ hOlO testz ~~~~~ hOlOw checkz ~~~~~~"
"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
test_phuz = True
if test_phuz:
    print('san tsouleriko')
    dphi = np.load(path_phase + "dphi.npy")  # ~~

    zator = normalize(dphi)

    dphi_err = np.load(path_phase + "dphi_err.npy")  # ~~
    a = (dphi_err.shape[0] // 2)
    # b=dphi_err[a:a+1, :]

    there_this = closest_arr(dphi_err[a:a+1, :][0], 620000.)

    plt.imshow(dphi_err[a:a+1, :], cmap='inferno')
    plt.colorbar()
    plt.title("dphi_err idx, {}".format(there_this))
    plt.show()

    missing = np.mean( [zator[a:a+1, there_this[0]-1], zator[a:a+1, there_this[0]+1]])
    print("l {}, r {}, m {}".format(zator[a:a+1, there_this[0]-1], zator[a:a+1, there_this[0]+1], missing))
    zator[a:a+1, there_this[0]] = missing

    circular_aperture = draw_circle(1024, 500)
    circular_aperture = center_overlay(1024, 1024, circular_aperture)

    plt.subplot(121)
    plt.imshow(dphi, cmap='inferno')
    plt.colorbar()
    plt.title("dphi")
    plt.subplot(122)
    plt.imshow(zator, cmap='inferno')
    plt.colorbar()
    plt.title("zator")
    plt.show()

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
    plt.imshow(resu_norm_mo, cmap='inferno')
    plt.title("resu_norm_mo")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()

    plt.subplot(131)
    plt.imshow(res_resz, cmap='inferno')
    plt.title("res_resz")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(132)
    plt.imshow(inv_zator_norm, cmap='inferno')
    plt.title("inv_zator_norm")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(133)
    plt.imshow(res_resz*circular_aperture, cmap='inferno')
    plt.title("aperitus")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()



print('es el finAl')
# 'es el finAl'

# plt.show(block=False)
# plt.pause(0.8)
# plt.close()
# if saVe_plo:
#     plt.show(block=False)
#     # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
#     figD.savefig(path_phase +'\\detPhuz.png', dpi=300, bbox_inches='tight', transparent=False)
#     plt.pause(2)
#     plt.close(figD)
# else:
#     plt.show()

