# this is an implementation of the method described in:
# https://doi.org/10.1016/j.optlaseng.2020.106132

# https://photutils.readthedocs.io/en/stable/centroids.html  # has script for centroid detection, a ha a a

import time
import os
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


save_path = "E:/mitsos/pYthOn/slm_chronicles/amphuz_retriev/result/phase_calibration/"
saVe_plot = True
if saVe_plot:
    date_saved = time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())
    save_path = save_path + date_saved
    if not os.path.exists(save_path):
        os.mkdir(save_path)

"init cam"
exp = 1
params = {'exposure': exp/1000, "initCam": True,
          "came_numb": 0, "trig_mODe": 1}

cam_obj = Cam.LiveHamamatsu(**params)
slm_disp_obj = slm

"prep cam"
frame_num = 1
cam_obj.mode = "Acq"
cam_obj.num = 1
cam_obj.bin_sz = 1
cam_obj.hcam.setACQMode('fixed_length', number_frames=cam_obj.num)
cam_obj.exposure

"roi"
cam_roi_pos = [0, 980]  # grat 10 [1230:1530, 1080:1380]
cam_roi_sz = [2048, 100]  # grat 10
cam_obj.roi_set_roi(int(cam_roi_pos[0] * cam_obj.bin_sz), int(cam_roi_pos[1] * cam_obj.bin_sz),
                    int(cam_roi_sz[0] * cam_obj.bin_sz), int(cam_roi_sz[1] * cam_obj.bin_sz))

"record background"
print(Fore.LIGHTGREEN_EX + "sha' record background" + Style.RESET_ALL)

"close shutter"
sh.shutter_state()
time.sleep(0.4)
if sh.shut_state == 1:
    sh.shutter_enable()
time.sleep(0.4)
sh.shutter_state()

time.sleep(0.5)

"set camera"
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

"prep phase basics"
phagen.whichphuzzez = {"grating": True, "lens": False, "phase": False, "amplitude": False, "corr_patt": True,
                       "lg_pol": False, "corr_phase": False}
phagen.linear_grating()

"run the loop"
# todo: can I replace the original take_average_image in the Cam Class with a new one that does
#  not prep every time

"phase"
phagen.modDepth = 220
phagen._make_full_slm_array()

"up to slm"
slm_disp_obj.display(phagen.final_phuz)

"get image"
cam_obj.take_average_image(frame_num)
img = cam_obj.last_frame
img_noBg = cam_obj.last_frame - bckgr

"stack em"
stack_sh_fk = np.concatenate((img_noBg, img, bckgr), axis=0)

"gotta keep em separated"
amps_primus, amps_nulla, amps_secundus = separate_graph_regions(stack_sh_fk, img_noBg,
                                                                save_path, modDepth=phagen.modDepth,
                                                                crop_sz=10, saVe_plo=saVe_plot)


# "show em"


# es el final
