import os
import numpy as np
import matplotlib.pyplot as plt
# https://indianaiproduction.com/image-to-video-opencv/
import cv2
from natsort import natsorted # pip install natsort


def center_overlay(bg_size_x, bg_size_y, arr):
    """
    ex - center_paste2canVas
    pastes an array to a zero valued background array of the shApe of the slm
    """
    arr = np.asarray(arr)
    aa = np.zeros((bg_size_y, bg_size_x))
    arr_width = arr.shape[1]
    arr_height = arr.shape[0]
    start_x = (bg_size_x - arr_width) // 2
    start_y = (bg_size_y - arr_height) // 2
    aa[start_y:start_y + arr_height, start_x:start_x + arr_width] = arr[:, :]
    return aa



curr_dir = os.getcwd()

path = ("E:\\mitsos\\pYthOn\\slm_chronicles\\amphuz_retriev\\amphase_result\\forVid\\"
        "24-06-02_18-05-46_measure_slm_wavefront\\")

prep_imgz = False
saVe_plo = True

if prep_imgz:
    mask_arr = []
    for num in range(900):
        fringe_path = path + "imgF_iter_{}.npy".format(num)
        mask_path = path + "masked_phase_iter_{}.npy".format(num)

        # im = Image.open(os.path.join(path, file))
        print(num)
        print(mask_path)
        fringe_arr = np.load(fringe_path)
        mask_arr = center_overlay(1272, 1272, np.load(mask_path))
        # np.concatenate((fringe_arr, mask_arr), axis=1)

        " Intensity Plot & fit ~"
        fig = plt.figure()
        plt.subplot(121), plt.imshow(fringe_arr, cmap='inferno', vmax=25, vmin=0)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(122), plt.imshow(mask_arr, cmap='inferno')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout(pad=0.0)  # plt.tight_layout(pad=5.0)
        path_sv = path + "vid_files\\"
        if saVe_plo:
            if not os.path.exists(path_sv):
                os.mkdir(path_sv)
            plt.show(block=False)
            fig.savefig(path_sv + "img_it_{}.png".format(num), dpi=300, bbox_inches='tight', pad_inches=0, transparent=False)
            # plt.pause(0.8)
            plt.close(fig)
        else:
            # plt.show()
            plt.show(block=False)
            plt.pause(0.4)
            plt.close(fig)

"mk video"
image_folder = path + "\\vid_files\\"

video_name = image_folder + 'video_jah.avi'

img_name_list = [img for img in os.listdir(image_folder) if img.endswith(".png")]

img_name_list = natsorted(img_name_list)
img_tes = cv2.imread(image_folder + img_name_list[0])
height, width, _ = img_tes.shape

fps = 10
sec = 90

fourcc = cv2.VideoWriter_fourcc(*'MP42')

video = cv2.VideoWriter(video_name, fourcc, float(fps), (width, height))

print("start render")
# print(img_name_list)
for frame_count in range(fps * sec):
    img_name = img_name_list[frame_count]
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img, (width, height))

    video.write(img_resize)

# cv2.destroyAllWindows()
video.release()

# es el finAl
