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

from experiment import Params#, Camera, SlmDisp

from colorama import Fore, Style  # , Back

pms_obj = Params()
slm_disp_obj = None
cam_obj = None
exp = 100
cam_roi_sz = [300, 300]


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
this_path = pms_obj.i_path

if load_existing:
    loaded_phuz = np.load(this_path)

    loPhuz = plt.figure()
    plt.imshow(loaded_phuz, cmap='turbo')
    # plt.imshow(loaded_phuz / np.pi / 2, cmap='magma')
    plt.colorbar()
    plt.title('intense')
    # plt.title('Unwrapped measured phase')

    # fig, axs = plt.subplots(1, 2)
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

    if saVe_plo:
        plt.show(block=False)
        # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
        loPhuz.savefig(this_path[:-9] +'\\int.png', dpi=300, bbox_inches='tight',
                    transparent=False)  # True trns worls nice for dispersion thinks I
        plt.pause(2.4)
        plt.close()
    else:
        plt.show()


    loaded_phuz = np.load(this_path)

    loPhuz = plt.figure()
    plt.imshow(loaded_phuz, cmap='turbo')
    # plt.imshow(loaded_phuz / np.pi / 2, cmap='magma')
    plt.colorbar()
    plt.title('intense')


print('es el finAl')
# 'es el finAl'
