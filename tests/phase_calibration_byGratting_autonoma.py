# track intensity of the 0th, 1st & 2nd order spots while changing the bit depth of the phase
# https://photutils.readthedocs.io/en/stable/centroids.html  # has script for centroid detection, a ha a a

import time
import os
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style  # , Back
import copy
from slm.helpers import separate_graph_regions
import orca.orca_autonoma as Cam
from slm.slm_hama_amphase import slm
from peripheral_instruments.thorlabs_shutter import shutter as sh
from slm.phase_generator_bOOm import phagen


"save conditions"
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
frame_num = 100
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
    plt.colorbar()
    plt.title('backg')
    plt.show()
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close(fig)

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

print("save condition is set to {}".format(saVe_plot))

amp_0 = []
amp_1st = []
amp_2nd = []
for mo in range(255):
    print("bit depth is {}".format(mo))

    "phase"
    phagen.modDepth = mo
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
    "book-keeping"
    amp_0.append(amps_nulla)
    amp_1st.append(amps_primus)
    amp_2nd.append(amps_secundus)

"save em"
np.save(save_path + '//amp_0', amp_0)
np.save(save_path + '//amp_1', amp_1st)
np.save(save_path + '//amp_2', amp_2nd)


# "show em"
Fig = plt.figure()
plt.subplot(131)
plt.plot(amp_1st[0][:], color='darkorchid', linestyle="dotted", linewidth=1.4, label="data")
plt.plot(amp_1st[1][:], color='teal', linewidth=0.8, label="fit")
# plt.title("amp_fit {}, amp_data {}".format(ampFit_primus, ampData_primus))
plt.title("amp_1st")
plt.legend()
plt.subplot(132)
plt.plot(amp_0[0][:], color='darkorchid', linestyle="dotted", linewidth=1.4, label="data")
plt.plot(amp_0[1][:], color='teal', linewidth=0.8, label="fit")
# plt.title("amp_fit {}, amp_data {}".format(ampFit_primus, ampData_primus))
plt.title("amp_0")
plt.legend()
plt.subplot(133)
plt.plot(amp_2nd[0][:], color='darkorchid', linestyle="dotted", linewidth=1.4, label="data")
plt.plot(amp_2nd[1][:], color='teal', linewidth=0.8, label="fit")
# plt.title("amp_fit {}, amp_data {}".format(ampFit_primus, ampData_primus))
plt.title("amp_2nd")
plt.legend()
plt.tight_layout()
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
if saVe_plot:
    plt.show(block=False)
    Fig.savefig(save_path + '\\bitDepth_calibration_result.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.pause(0.8)
    plt.close(Fig)
else:
    plt.show()

# es el final
