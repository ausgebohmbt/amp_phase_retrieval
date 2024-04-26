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
# import matplotlib.pyplot as plt
import calibrate_slm as clb

from experiment import Params, Camera, SlmDisp

import gui_standalone.CameraInterfaces_ising as Cam
from gui_standalone.hamamatsu_camera import n_cameras as numb_cam
from slm.slm_hama_4gui import slm


pms_obj = Params()
cam_obj = Camera(np.array([960, 1280]), 3.75e-6, bayer=True)    # fixme: pitch used for cam size
slm_disp_obj = SlmDisp(np.array([1024, 1280]), 12.5e-6)         # fixme: pitch used meshgrid

measure_slm_intensity = True   # Measure the constant intensity at the SLM (laser beam profile)?
measure_slm_phase = True       # Measure the constant phase at the SLM?

"Measuring the constant intensity and phase at the SLM"
if measure_slm_intensity is True:
    i_path = clb.measure_slm_intensity(slm_disp_obj, cam_obj, pms_obj, 30, 32, 10000, 256, 300)
    pms_obj.i_path = i_path
if measure_slm_phase is True:
    phi_path = clb.measure_slm_wavefront(slm_disp_obj, cam_obj, pms_obj, 30, 16, 64, 40000, 256, roi_min_x=2,
                                         roi_min_y=2, roi_n=26)
    pms_obj.phi_path = phi_path

print('es el finAl')
# 'es el finAl'
