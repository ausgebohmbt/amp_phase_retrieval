""""creates a list of radians the size of the slm modulation depth"""

import numpy as np
import math


class Rad2Bitmap:
    def __init__(self):
        self.rad_list = list
        self.slm_mod_depth = 220
        self.phase_map = range(self.slm_mod_depth + 1)
        self.increment = 360/self.slm_mod_depth
        self.degrz = [0.]
        self.radi_calc = np.zeros(self.slm_mod_depth + 1)

    def convert_rad2uint(self):
        """does as says"""
        for ha in range(self.slm_mod_depth):
            self.degrz.append(self.degrz[ha] + self.increment)
        for haha in range(len(self.degrz)):
            self.radi_calc[haha] = math.radians(self.degrz[haha])

    def closest(self, lst, K) -> tuple:
        """searches list lst for the closest value to K.

         return:
             tuple with index and closest value existing the list"""
        return min(enumerate(lst), key=lambda x: abs(x[1]-K))


radical = Rad2Bitmap()
radical.convert_rad2uint()
print("0 and 220 of aaaha {} & {}".format(radical.degrz[0], radical.degrz[radical.slm_mod_depth]))
print("0 and 220 of aaaha {} & {}".format(radical.radi_calc[0], radical.radi_calc[radical.slm_mod_depth]))
print(radical.closest(radical.radi_calc, 2*np.pi))

# es el finAl
