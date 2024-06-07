""""creates a list of radians the size of the slm modulation depth"""

import numpy as np
import math


def closest(lst, K) -> tuple:
    """searches list lst for the closest value to K.

     return:
         tuple with index and closest value existing the list"""
    return min(enumerate(lst), key=lambda x: abs(x[1]-K))


slm_mod_depth = 220
phase_map = range(slm_mod_depth + 1)
increment = 360/slm_mod_depth
print(increment)

degrz = [0]
radi_calc = np.zeros(slm_mod_depth + 1)
for ha in range(slm_mod_depth):
    degrz.append(degrz[ha] + increment)
    print("ha {} val {}".format(ha, degrz[ha]))
for haha in range(len(degrz)):
    radi_calc[haha] = math.radians(degrz[haha])
    print("radi_calc val {}".format(radi_calc[haha]))

print("0 and 220 of aaaha {} & {}".format(degrz[0], degrz[slm_mod_depth]))
print("0 and 220 of aaaha {} & {}".format(radi_calc[0], radi_calc[slm_mod_depth]))
print(math.radians(359.99))
print(closest(radi_calc, np.pi))

# es el finAl
