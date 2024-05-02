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
import calibrate_slm as clb

from experiment import Params#, Camera, SlmDisp

import orca.orca_autonoma as Cam
# from orca.hamamatsu_camera import n_cameras as numb_cam
from slm.slm_hama_amphase import slm
from slm.phase_generator import phagen as phuzGen
from peripheral_instruments.thorlabs_shutter import shutter as sh
from colorama import Fore, Style  # , Back

exp = 1500
params = {'exposure': exp/1000, "initCam": True,
          "came_numb": 0, "trig_mODe": 1}
# phuzGen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": True}
# phuzGen.linear_grating()

pms_obj = Params()
cam_obj = Cam.LiveHamamatsu(**params)
slm_disp_obj = slm
# cam_obj = Camera(np.array([960, 1280]), 3.75e-6, bayer=True)    # fixme: pitch used for cam size
# slm_disp_obj = SlmDisp(np.array([1024, 1280]), 12.5e-6)         # fixme: pitch used meshgrid

"upload phuz"
# slm.current_phase = phuzGen.final_phuz
# slm.load_phase(slm.current_phase)

"init cam"
cam_obj.mode = "Acq"
cam_obj.num = 1
cam_obj.bin_sz = 1
# cam_roi_pos = (936, 942)
# cam_roi_sz = (788, 182)
cam_roi_pos = [175, 1400]
cam_roi_sz = [300, 300]
# cam_obj.roi_set_roi(int(cam_roi_pos[0] * cam_obj.bin_sz), int(cam_roi_pos[1] * cam_obj.bin_sz),
#                     int(cam_roi_sz[0] * cam_obj.bin_sz), int(cam_roi_sz[1] * cam_obj.bin_sz))

# "take test img"
# cam_obj.roi_set_roi(int(cam_roi_pos[0] * cam_obj.bin_sz), int(cam_roi_pos[1] * cam_obj.bin_sz),
#                     int(cam_roi_sz[0] * cam_obj.bin_sz), int(cam_roi_sz[1] * cam_obj.bin_sz))
cam_obj.hcam.setACQMode('fixed_length', number_frames=cam_obj.num)
# cam_obj.take_image()
# imgzaz = cam_obj.last_frame
#
# fig = plt.figure()
# plt.imshow(imgzaz, cmap='inferno', vmax=1000)
# plt.colorbar()
# plt.show()

measure_slm_intensity = False   # Measure the constant intensity at the SLM (laser beam profile)?
measure_slm_phase = True       # Measure the constant phase at the SLM?

"Measuring the constant intensity and phase at the SLM"
if measure_slm_intensity is True:
    # i_path = clb.measure_slm_intensity(slm_disp_obj, cam_obj, pms_obj,
    #                                    30, 32, 10000,
    #                                    256, np.asarray(cam_roi_sz[0]))

    i_path = clb.measure_slm_intensity(slm_disp_obj, cam_obj, pms_obj,
                                       30, 32, exp/1000,
                                       256, np.asarray(cam_roi_sz[0]))
    pms_obj.i_path = i_path
if measure_slm_phase is True:
    # phi_path = clb.measure_slm_wavefront(slm_disp_obj, cam_obj, pms_obj, 30, 16, 64, 40000, 256, roi_min_x=2,
    #                                      roi_min_y=2, roi_n=26)
    phi_path = clb.measure_slm_wavefront(slm_disp_obj, cam_obj, pms_obj, 30, 16,
                                         64, 40000, 256, n_avg_frames=5, roi_min_x=0,
                                         roi_min_y=0, roi_n=30)
    pms_obj.phi_path = phi_path


cam_obj.end()

load_existing = False
saVe_plo = True
# this_path = pms_obj.phi_path
this_path = pms_obj.i_path

if load_existing:
    loaded_phuz = np.load(this_path)

    loPhuz = plt.figure()
    plt.imshow(loaded_phuz, cmap='magma')
    # plt.imshow(loaded_phuz / np.pi / 2, cmap='magma')
    plt.colorbar()
    plt.title('intense')
    # plt.title('Unwrapped measured phase')
    if saVe_plo:
        plt.show(block=False)
        # img_nm = img_nom[:-4].replace(data_pAth_ame, '')meas_nom
        loPhuz.savefig(this_path[:-9] +'\\int.png', dpi=300, bbox_inches='tight',
                    transparent=False)  # True trns worls nice for dispersion thinks I
        plt.pause(2.4)
        plt.close()
    else:
        plt.show()



print('es el finAl')
# 'es el finAl'
