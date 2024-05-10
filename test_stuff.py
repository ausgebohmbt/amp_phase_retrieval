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
# import calibrate_slm as clb
import measurement_functions as mfunc
import error_metrics as m, patterns as pt, fitting as ft

from experiment import Params#, Camera, SlmDisp

from colorama import Fore, Style  # , Back

pms_obj = Params()
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
saVe_plo = False
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
main_path = "E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\amphase_result\\24-05-10_17-30-45_measure_slm_wavefront\\"

img = np.load(main_path + "img.npy")
popt_sv = np.load(main_path + "popt_sv.npy")
powah = np.load(main_path + "powah.npy")
power = np.load(main_path + "powah.npy")
i_fit = np.load(main_path + "i_fit.npy")

plt.imshow(img[:, :, -1], cmap='turbo')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title("img[:, :, -1]")
plt.show()

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
    fig1.savefig(main_path + '\\fit.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.pause(2)
    plt.close(fig1)
else:
    plt.show()

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

figPow = plt.Figure()
plt.subplot(121)
plt.imshow(powah, cmap='turbo')
plt.title("powah")
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(122)
plt.imshow(power, cmap='turbo')
plt.title("powah")
plt.colorbar(fraction=0.046, pad=0.04)
if saVe_plo:
    plt.show(block=False)
    # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
    figPow.savefig(main_path +'\\powaher.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.pause(2)
    plt.close(figPow)
else:
    plt.show()


print('es el finAl')
# 'es el finAl'
