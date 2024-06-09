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
import calibrate_slm as clb
import matplotlib.pyplot as plt
from colorama import Fore, Style  # , Back

from experiment import Params

import orca.orca_autonoma as Cam
from slm.slm_hama_amphase import slm
from slm.helpers import normalize, unimOD, closest_arr
from slm.phase_generator_bOOm import phagen


exp = 310
params = {'exposure': exp/1000, "initCam": True,
          "came_numb": 0, "trig_mODe": 1}

pms_obj = Params()
cam_obj = Cam.LiveHamamatsu(**params)
slm_disp_obj = slm

"init cam"
cam_obj.mode = "Acq"
cam_obj.num = 1
cam_obj.bin_sz = 1
cam_roi_pos = [175, 1400]
cam_roi_sz = [300, 300]

cam_obj.hcam.setACQMode('fixed_length', number_frames=cam_obj.num)
cam_obj.exposure

measure_slm_intensity = False   # Measure the constant intensity at the SLM (laser beam profile)?
measure_slm_phase = False       # Measure the constant phase at the SLM?
correction_section = True

"Measuring the constant intensity and phase at the SLM"
if measure_slm_intensity is True:
    i_path = clb.measure_slm_intensity(slm_disp_obj, cam_obj, pms_obj,
                                       30, 32, exp/1000,
                                       256, np.asarray(cam_roi_sz[0]))
    pms_obj.i_path = i_path
if measure_slm_phase is True:
    cam_obj.exposure
    phi_path = clb.measure_slm_wavefront(slm_disp_obj, cam_obj, pms_obj, 30, 16,
                                         64, exp/1000, 256, n_avg_frames=5, roi_min_x=0,
                                         roi_min_y=0, roi_n=30)

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
" create correction phase mask and apply it to correct LG order 2 ~~~~~~~"
"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
if not measure_slm_phase:
    path_phase = ("E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\"
                  "amphase_result\\24-06-07_12-08-15_measure_slm_wavefront\\")
else:
    path_phase = phi_path

correction_section = True
if correction_section:
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

    figD = plt.Figure()
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

    "cre'te spiral phase and start testin"
    phagen.laguerre_beam(r0=(0, 0), A=1, order_rdl=0, order_ang=1, z=0.01, z0=0)


print('es el finAl')






cam_obj.end()
cam_obj.hcam.shutdown()

print(Fore.LIGHTBLUE_EX + "jah ha run" + Style.RESET_ALL)
print('es el finAl')
# 'es el finAl'
