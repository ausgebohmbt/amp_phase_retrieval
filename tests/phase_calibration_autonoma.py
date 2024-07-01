# https://photutils.readthedocs.io/en/stable/centroids.html  # has script for centroid detection, a ha a a

import matplotlib.pyplot as plt
from slm.helpers import (normalize, unimOD, closest_arr, draw_circle, center_overlay,
                         draw_circle_displaced,  draw_n_paste_circle)

radios = 34
singcular_aperture = draw_circle_displaced(1024, radios, 250, 0)
circular_aperture = draw_n_paste_circle(singcular_aperture*255, radios, -250, 0, 42)
circular_aperture = center_overlay(1272, 1024, circular_aperture)


plt.subplot(121)
plt.imshow(singcular_aperture, cmap='inferno')
plt.title("res_resz")
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(122)
plt.imshow(circular_aperture, cmap='inferno')
plt.title("inv_zator_norm")
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
