# this is an implementation of the method described in:
# https://doi.org/10.1016/j.optlaseng.2020.106132

# https://photutils.readthedocs.io/en/stable/centroids.html  # has script for centroid detection, a ha a a

import time
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style  # , Back
import copy
from slm.helpers import (normalize, unimOD, closest_arr, draw_circle, center_overlay,
                         draw_circle_displaced,  draw_n_paste_circle, separate_graph_regions)
import orca.orca_autonoma as Cam
from slm.slm_hama_amphase import slm
from peripheral_instruments.thorlabs_shutter import shutter as sh
from slm.phase_generator_bOOm import phagen

"phase"
phagen.modDepth = 220
phagen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": True,
                       "lg_pol": False, "corr_phase": False}
phagen.linear_grating()
phagen._make_full_slm_array()
slm_phase = phagen.final_phuz

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
cam_roi_pos = [0, 980]  # grat 10 [1230:1530, 1080:1380]
cam_roi_sz = [2048, 100]  # grat 10
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

time.sleep(0.5)

frame_num = 1
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

"stack em"
stack_sh_fk = np.concatenate((img_noBg, img, bckgr), axis=0)

plo_che = False
if plo_che:
    fig = plt.figure()
    plt.subplot(221), plt.imshow(img_noBg, cmap='inferno', vmin=0, vmax=1450)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('no bckgr: {}'.format(4))
    plt.subplot(222), plt.imshow(img, cmap='inferno', vmax=2600)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('original img: {}'.format(2))
    plt.subplot(223), plt.imshow(bckgr, cmap='inferno', vmax=600)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('bckgr')
    plt.subplot(224), plt.imshow(stack_sh_fk, cmap='inferno', vmax=600)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('stack_sh_fk')
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

slm_disp_obj.display(slm_phase)
cam_obj.take_average_image(frame_num)
imgzaz = cam_obj.last_frame - bckgr

"gotta keep em separated"
stack_primus = separate_graph_regions(stack_sh_fk, img_noBg)

"show em"
rezs = plt.figure()
plt.subplot(231)
plt.imshow(np.log10(stack_sh_fk), cmap='inferno')
plt.title("log o' stack")
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(232)
plt.plot(stack_sh_fk[:, 446])
plt.plot(stack_sh_fk[:, 1010])
plt.plot(stack_sh_fk[:, 1568])
plt.title("profs")
plt.legend(["2nd order", "1st order", "0th order"])
plt.subplot(233)
plt.plot(imgzaz_tipota[50, :])
plt.plot(imgzaz[50, :])
plt.title("profile at {}".format(50))
plt.legend(["empty phase", "corretion & grating"])
plt.subplot(234)
plt.plot(imgzaz[50, :] - imgzaz_tipota[50, :])
plt.title("diff of prof @ {}".format(50))
plt.subplot(235)
plt.imshow(nossing, cmap='inferno')
plt.title("nossing")
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(236)
plt.imshow(stack_primus, cmap='inferno')
# plt.imshow(slm_phase, cmap='inferno')
plt.title("phase, bitness {}".format(phagen.modDepth))
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

# es el final
