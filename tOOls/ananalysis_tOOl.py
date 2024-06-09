"""
Script to  visualise and analyze the results of the Feedback Algorithm Example
==========================
"""

import numpy as np
import matplotlib.pyplot as plt
import os
# from matplotlib import cm
# from colorama import Fore, Style  # , Back

# import measurement_functions as mfunc
import error_metrics as m, patterns as pt, fitting as ft
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from slm.phase_generator import phagen as phuzGen
from slm.helpers import normalize  # , unimOD, center_overlay  # , center_crop, tiler,
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
    plt.imshow(dphi, cmap='inferno')
    plt.colorbar()
    plt.title("dphi")
    plt.show()

    zator = normalize(dphi)

    plt.imshow(zator, cmap='inferno')
    plt.colorbar()
    plt.title("zator")
    plt.show()



    # npix = 1024
    # res_y, res_x = 1024, 1272
    # border_x = int(np.abs(res_x - res_y) // 2)
    #
    # spot_pos = 256
    # aperture_number = 30
    # aperture_width = 32
    #
    # zeros = np.zeros((npix, npix))
    # zeros_full = np.zeros((npix, 1272))
    #
    # lin_phase = np.array([-spot_pos, -spot_pos])
    # slm_phase4me = mfunc.init_pha(np.zeros((1024, 1272)), 1272, 12.5e-6,
    #                      pms_obj, lin_phase=lin_phase)
    # # slm_phaseOUT = pt.init_phase(np.zeros((aperture_width, aperture_width)), slm_disp_obj,
    # #                      pms_obj, lin_phase=lin_phase)
    # linFlip = np.fliplr(slm_phase4me)
    #
    # # slm_phase = pt.init_phase(np.zeros((aperture_width, aperture_width)), slm_disp_obj, pms_obj, lin_phase=lin_phase)
    # # slm_phase = np.remainder(slm_phaseOUT, 2 * np.pi)
    # slm_phase4me = np.remainder(slm_phase4me, 2 * np.pi)
    # slm_phase4me = np.fliplr(slm_phase4me)
    #
    #
    # # slm_phase = np.remainder(slm_phase, 2 * np.pi)
    # slm_phase = slm_phase4me
    # # slm_phase = np.flipud(np.fliplr(slm_phase))
    # slm_idx = mfunc.get_aperture_indices(aperture_number, aperture_number, border_x, npix + border_x, 0, npix, aperture_width,
    #                                aperture_width)
    #
    # # Display central sub-aperture on SLM and check if camera is over-exposed.
    # i = (aperture_number ** 2) // 2 - aperture_number // 2
    # phi_centre = np.zeros_like(zeros)
    # # phi_centre[slm_idx[0][i]:slm_idx[1][i], slm_idx[2][i]:slm_idx[3][i]] = slm_phase
    # phi_centre = slm_phase
    # slm_phaseNOR = mfunc.normalize(slm_phase)
    #
    # # plt.imshow(slm_phaseNOR, cmap='inferno')
    # # plt.colorbar()
    # # plt.title("slm_phaseNOR")
    # # plt.show()
    #
    # phuzGen.diviX = 10
    # phuzGen.diviY = 10
    # phuzGen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": True}
    # # phuzGen.linear_grating()
    # phuzGen.grat = slm_phaseNOR
    # phuzGen._make_full_slm_array()
    # phi_centre = phuzGen.final_phuz
    # phi_centre = mfunc.normalize(phi_centre)*220
    #
    # lg = center_overlay(res_x, res_y, phuzGen.lgPhu_mod)
    # lgra = np.add(phuzGen.grat, lg)
    # phuz = unimOD(lgra) * phuzGen.modDepth
    # lg_gra = phuz.astype('uint8')
    #
    # roi_min_x = 0
    # roi_min_y = 0
    # roi_n = 30
    # # Determine phase
    # dphi_uw_nopa = pt.unwrap_2d(lg_gra)
    # dphi_uw_notil = ft.remove_tilt(dphi_uw_nopa)
    # pad_ro = ((roi_min_x, aperture_number - roi_n - roi_min_x), (roi_min_y, aperture_number - roi_n - roi_min_y))
    # dph_uw = np.pad(dphi_uw_nopa, pad_ro)
    # # dph_uw = mfunc.normalize(dph_uw)
    # mumnwra = dph_uw / np.pi / 2
    #
    # figD = plt.Figure()
    # plt.subplot(231)
    # plt.imshow(phuzGen.lgPhu_mod, cmap='inferno')
    # plt.title("lg")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.subplot(232)
    # plt.imshow(dphi_uw_notil, cmap='inferno')
    # plt.title("rm piston tilt")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.subplot(233)
    # plt.imshow(phuzGen.grat, cmap='inferno')
    # plt.title("grat")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.subplot(234)
    # plt.imshow(lg_gra, cmap='inferno')
    # plt.title("lg_gra")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.subplot(235)
    # plt.imshow(linFlip, cmap='inferno')
    # plt.title("linFlip")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.subplot(236)
    # plt.imshow(mumnwra, cmap='inferno')
    # plt.title("mumnwra")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # # plt.show()
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

print('es el finAl')
# 'es el finAl'
