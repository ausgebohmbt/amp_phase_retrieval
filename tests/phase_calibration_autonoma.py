# this is an implementation of the method described in:
# https://doi.org/10.1016/j.optlaseng.2020.106132

# https://photutils.readthedocs.io/en/stable/centroids.html  # has script for centroid detection, a ha a a

import time
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style  # , Back
import copy
from slm.helpers import (normalize, unimOD, closest_arr, draw_circle, center_overlay,
                         draw_circle_displaced,  draw_n_paste_circle)
import orca.orca_autonoma as Cam
from slm.slm_hama_amphase import slm
from peripheral_instruments.thorlabs_shutter import shutter as sh
from slm.phase_generator_bOOm import phagen

radios = 31
spacing = 200
more_space = 0
print("spacing {}".format(spacing))
singcular_aperture = draw_circle_displaced(1024, radios, spacing + more_space, 0)
her_bitness = 120
circular_aperture = draw_n_paste_circle(singcular_aperture*120, radios, -(spacing + more_space),
                                        0, her_bitness)
circular_aperture = center_overlay(1272, 1024, circular_aperture)

"empty phases"
# phagen.state_lg = False
# phagen.whichphuzzez = {"grating": False, "lens": False, "phase": False, "amplitude": False, "corr_patt": True,
#                        "lg_pol": False, "corr_phase": False}
# # phagen.linear_grating()
# phagen._make_full_slm_array()
# nossing = phagen.final_phuz

"circles a ha a a"
# phagen.crxn_phuz_boom = circular_aperture
# phagen.whichphuzzez = {"grating": False, "lens": False, "phase": False, "amplitude": False, "corr_patt": True,
#                        "lg_pol": False, "corr_phase": True}
# # phagen.linear_grating()
# phagen._make_full_slm_array()
# slm_phase_init = phagen.final_phuz

# fi = plt.figure()
# plt.subplot(121)
# plt.imshow(slm_phase_init, cmap='inferno')
# plt.title("res_resz")
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.subplot(122)
# plt.imshow(circular_aperture, cmap='inferno')
# plt.title("inv_zator_norm")
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.show(block=False)
# # fig.savefig(path + '\\iter_{}'.format(i) + '_full.png', dpi=300, bbox_inches='tight', transparent=False)
# # Save data
# # np.save(path + '\\imgF_iter_{}'.format(i), imgF)
# plt.pause(0.8)
# plt.close(fi)

"init cam"
exp = 1
params = {'exposure': exp/1000, "initCam": True,
          "came_numb": 0, "trig_mODe": 1}

cam_obj = Cam.LiveHamamatsu(**params)
slm_disp_obj = slm

"prep cam"
cam_obj.mode = "Acq"
cam_obj.num = 1
cam_obj.bin_sz = 1
# cam_roi_pos = [175, 1400]
# cam_roi_sz = [300, 300]

cam_obj.hcam.setACQMode('fixed_length', number_frames=cam_obj.num)
cam_obj.exposure

"roi"
cam_roi_pos = [1300, 800]  # grat 10 [1230:1530, 1080:1380]
cam_roi_sz = [500, 500]  # grat 10
cam_obj.roi_set_roi(int(cam_roi_pos[0] * cam_obj.bin_sz), int(cam_roi_pos[1] * cam_obj.bin_sz),
                    int(cam_roi_sz[0] * cam_obj.bin_sz), int(cam_roi_sz[1] * cam_obj.bin_sz))

"record background"

print(Fore.LIGHTGREEN_EX + "record background" + Style.RESET_ALL)

"close shutter"
sh.shutter_state()
time.sleep(0.4)
if sh.shut_state == 1:
    sh.shutter_enable()
time.sleep(0.4)
sh.shutter_state()

# time.sleep(1)

frame_num = 20
cam_obj.stop_acq()
cam_obj.take_average_image(frame_num)
cam_obj.bckgr = copy.deepcopy(cam_obj.last_frame)
bckgr = copy.deepcopy(cam_obj.bckgr)

plo_che = False
if plo_che:
    fig = plt.figure()
    plt.imshow(bckgr, cmap='inferno', vmax=150)
    # plt.imshow(bckgr[1230:1530, 1080:1380], cmap='inferno', vmax=150)
    plt.colorbar()
    plt.title('backg')
    plt.show()
    # plt.show(block=False)
    plt.pause(1)
    plt.close(fig)

"open shutter"
sh.shutter_state()
time.sleep(0.1)
if sh.shut_state == 0:
    sh.shutter_enable()
time.sleep(0.4)
sh.shutter_state()

cam_obj.take_average_image(frame_num)
img = cam_obj.last_frame
img_noBg = cam_obj.last_frame - bckgr


plo_che = False
if plo_che:
    fig = plt.figure()
    plt.subplot(131), plt.imshow(img_noBg, cmap='inferno', vmin=0, vmax=1450)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('no bckgr: {}'.format(4))
    plt.subplot(132), plt.imshow(img, cmap='inferno')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('original img: {}'.format(2))
    # plt.subplot(133), plt.imshow(img[..., i], cmap='inferno', vmax=600)
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.title('ROI')
    plt.show()
    # plt.show(block=False)
    # fig.savefig(path + '\\iter_{}'.format(i) + '_full.png', dpi=300, bbox_inches='tight', transparent=False)
    # Save data
    # np.save(path + '\\imgF_iter_{}'.format(i), imgF)
    plt.pause(0.8)
    plt.close(fig)

"run the loop"
# todo: can I replace the original take_average_image in the Cam Class with a new one that does
#  not prep every time

nossing = np.zeros((1024, 1272))


"upload blank 2 slm"
slm_disp_obj.display(nossing)
cam_obj.take_average_image(frame_num)
imgzaz_tipota = cam_obj.last_frame - bckgr

slm_disp_obj.display(circular_aperture)
cam_obj.take_average_image(frame_num)
imgzaz = cam_obj.last_frame - bckgr


plt.subplot(231)
plt.imshow(np.log10(imgzaz_tipota), cmap='inferno', vmax=3)
plt.title("imgzaz nothing")
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(232)
plt.imshow(np.log10(imgzaz - imgzaz_tipota), cmap='inferno')
plt.title("imgzaz clear log")
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(233)
plt.imshow(circular_aperture, cmap='inferno')
plt.title("slm_phase_init")
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(234)
plt.plot(imgzaz_tipota[227, :])
plt.plot(imgzaz[227, :])
plt.title("profile at {}".format(227))
plt.subplot(235)
plt.imshow(nossing - circular_aperture, cmap='inferno')
plt.title("nossing")
plt.subplot(236)
plt.plot(imgzaz_tipota[100, :] - imgzaz[100, :])
plt.title("log10 of profile at {}".format(100))
plt.show()
