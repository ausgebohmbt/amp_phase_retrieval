# loads and handles data from the phase_calibration_byGrating_autonoma.py script

import time
import os
import matplotlib.pyplot as plt
import numpy as np
# from colorama import Fore, Style  # , Back
# import copy


"paths"
load_path = "E:/mitsos/pYthOn/slm_chronicles/amphuz_retriev/result/phase_calibration/24-07-04_17-56-29"
save_path = load_path + 'analysis'
saVe_plots = True
if saVe_plots:
    if not os.path.exists(load_path):
        os.mkdir(load_path)

"load em"
amp_0 = np.load(load_path + '//amp_0.npy')
amp_1st = np.load(load_path + '//amp_1.npy')
amp_2nd = np.load(load_path + '//amp_2.npy')

"read em"
amp_0th_data = []
amp_0th_fit = []
for i in amp_0:
    amp_0th_data.append(i[0])
    amp_0th_fit.append(i[1])
amp_1st_data = []
amp_1st_fit = []
for i in amp_1st:
    amp_1st_data.append(i[0])
    amp_1st_fit.append(i[1])
amp_2nd_data = []
amp_2nd_fit = []
for i in amp_2nd:
    amp_2nd_data.append(i[0])
    amp_2nd_fit.append(i[1])

"get info"
if np.argmax(amp_1st_data) > np.argmax(amp_1st_fit):
    lim_o_y_1 = amp_1st_data[np.argmax(amp_1st_data)] + 444
else:
    lim_o_y_1 = amp_1st_fit[np.argmax(amp_1st_fit)] + 444
if np.argmax(amp_0th_data) > np.argmax(amp_0th_fit):
    lim_o_y_0 = amp_0th_data[np.argmax(amp_0th_data)] + 444
else:
    lim_o_y_0 = amp_0th_fit[np.argmax(amp_0th_fit)] + 444
if np.argmax(amp_2nd_data) > np.argmax(amp_2nd_fit):
    lim_o_y_2 = amp_2nd_data[np.argmax(amp_2nd_data)] + 444
else:
    lim_o_y_2 = amp_2nd_fit[np.argmax(amp_2nd_fit)] + 444


# "show em"
Fig = plt.figure()
plt.subplot(221)
plt.plot(amp_1st_data, color='darkorchid', linestyle="dotted", linewidth=1.4, label="data")
plt.plot(amp_1st_fit, color='teal', linewidth=0.8, label="fit")
plt.vlines(np.argmax(amp_1st_data), colors='darkorchid', linewidth=1.2, linestyle="dotted", label="data max",
           ymin=0, ymax=lim_o_y_1)
plt.vlines(np.argmax(amp_1st_fit), colors='teal', linewidth=0.8, linestyle="dashdot", label="fit max",
           ymin=0, ymax=(lim_o_y_1))
plt.ylim(0, lim_o_y_1)
plt.xlim(0, 255)
plt.title("1st order: data max {}, fit max {}".format(np.argmax(amp_1st_data), np.argmax(amp_1st_fit)),
          fontsize=10)
plt.legend()
plt.subplot(222)
plt.plot(amp_0th_data, color='darkorchid', linestyle="dotted", linewidth=1.4, label="data")
plt.plot(amp_0th_fit, color='teal', linewidth=0.8, label="fit")
plt.vlines(np.argmin(amp_0th_data), colors='darkorchid', linewidth=1.2, linestyle="dotted", label="data min",
           ymin=0, ymax=lim_o_y_0)
plt.vlines(np.argmin(amp_0th_fit), colors='teal', linewidth=0.8, linestyle="dashdot", label="fit min",
           ymin=0, ymax=(lim_o_y_0))
plt.ylim(0, lim_o_y_0)
plt.xlim(0, 255)
plt.title("0th order: data min {}, fit min {}".format(np.argmin(amp_0th_data), np.argmin(amp_0th_fit)),
          fontsize=10)
plt.legend()
plt.subplot(223)
plt.plot(amp_2nd_data, color='darkorchid', linestyle="dotted", linewidth=1.4, label="data")
plt.plot(amp_2nd_fit, color='teal', linewidth=0.8, label="fit")
plt.vlines(np.argmax(amp_2nd_data), colors='darkorchid', linewidth=1.2, linestyle="dotted", label="data max",
           ymin=0, ymax=lim_o_y_2)
plt.vlines(np.argmax(amp_2nd_fit), colors='teal', linewidth=0.8, linestyle="dashdot", label="fit max",
           ymin=0, ymax=(lim_o_y_2))
plt.ylim(0, lim_o_y_2)
plt.xlim(0, 255)
plt.title("2nd order: data max {}, fit max {}".format(np.argmax(amp_2nd_data), np.argmax(amp_2nd_fit)),
          fontsize=10)
plt.legend()
plt.tight_layout()
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
if saVe_plots:
    plt.show(block=False)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    Fig.savefig(load_path + '\\bitDepth_calibration_result.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.pause(0.8)
    plt.close(Fig)
else:
    plt.show()

# es el final
