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

from experiment import Params#, Camera, SlmDisp

import orca.orca_autonoma as Cam
from slm.slm_hama_amphase import slm
from colorama import Fore, Style  # , Back

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
measure_slm_phase = True       # Measure the constant phase at the SLM?

"Measuring the constant intensity and phase at the SLM"
if measure_slm_intensity is True:
    i_path = clb.measure_slm_intensity(slm_disp_obj, cam_obj, pms_obj,
                                       30, 32, exp/1000,
                                       256, np.asarray(cam_roi_sz[0]))
    # (slm_disp_obj, cam_obj, pms_obj, 30, 32, 10000, 256, np.asarray(cam_roi_sz[0]))
    pms_obj.i_path = i_path
if measure_slm_phase is True:
    cam_obj.exposure
    phi_path = clb.measure_slm_wavefront(slm_disp_obj, cam_obj, pms_obj, 30, 16,
                                         64, exp/1000, 256, n_avg_frames=5, roi_min_x=0,
                                         roi_min_y=0, roi_n=30)
    # (slm_disp_obj, cam_obj, pms_obj, 30, 16, 64, 40000, 256, roi_min_x=2, roi_min_y=2, roi_n=26)
    pms_obj.phi_path = phi_path

cam_obj.end()
cam_obj.hcam.shutdown()

print(Fore.LIGHTBLUE_EX + "jah ha run" + Style.RESET_ALL)
print('es el finAl')
# 'es el finAl'
