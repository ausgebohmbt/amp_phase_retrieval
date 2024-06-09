import os
import time

from cffi import FFI
import numpy as np
# import glob
# from PIL import Image
# from ..ising_machine_base.helpers_4gui import unimOD, normalize, center_overlay
# import matplotlib.pyplot as plt

# TODO: where to init, n main?
# TODO: spin_to_phase .. ,,, ,,, ,,,
# TODO: move unimod, normalize, etc to helper script


import abc
# from PyQt5 import QtCore


class SLM_clash():
    """Base class for all Spatial Light Modulators."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, parent=None):
        super().__init__()

    @abc.abstractmethod
    def load_phase(self, phase: np.ndarray) -> None:
        """Load a phase array onto the SLM display

        Returns
        -------
            not a thing
        """
        pass

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def end(self):
        pass


# Communicate with hamamatsu slm
class SlmDisplay(SLM_clash):
    """this is the class for the x15213 lcos-slm of hamamatsu"""

    def __init__(self, parent=None):

        super(SlmDisplay, self).__init__(parent)
        """ slm init variables"""
        cur_path = os.getcwd() + r"\slm"
        if "tests" in cur_path:
            cur_path = cur_path.replace("\\tests", "")
        SLM_DLL_FOLDER = cur_path + r"\hpkSLMdaLV_stdcall_64bit"
        os.environ['PATH'] = SLM_DLL_FOLDER + os.pathsep + os.environ.get('PATH', '')
        os.add_dll_directory(cur_path + r"\hpkSLMdaLV_stdcall_64bit")

        self.ffi = FFI()
        self.cdef_from_file = None
        self.header = cur_path + r"\hpkSLMdaLV_stdcall_64bit\hpkSLMdaLVt.h"
        self.slmffi = self.ffi.dlopen(cur_path + r"\hpkSLMdaLV_stdcall_64bit\hpkSLMdaLV.dll")
        self.bID = []
        # """ phase functions init variables"""
        self.slmX = 1272
        self.slmY = 1024
        self.pitch = 12.5e-6
        self.res = [self.slmY, self.slmX]
        self.current_phase = np.zeros((self.slmY, self.slmX))
        self.slm_size = self.pitch * np.asarray(self.res)

    def connect(self) -> int:
        """opens communication with the slm

        Returns
        -------
        int & sets upon self
            writes bID to the class variables and returns it as a string
        """

        try:
            with open(self.header) as slm_header:
                self.cdef_from_file = slm_header.read()

        except FileNotFoundError:
            return ("header not found")
        except IOError:
            return ("could not open header")
        finally:
            if self.cdef_from_file == '':
                return ('empty header')

        self.ffi.cdef(self.cdef_from_file, override=True)
        bIDList = self.ffi.new('uint8_t[10]')
        bIDSize = self.ffi.new('int32_t *')
        try:
            devNumber = self.slmffi.Open_Dev(bIDList, 10)
            print('slm init: com ok')
        except:
                print('slm init: no communication')
        self.bID = bIDList[0]
        # print("bid")
        # print(bIDList)
        return self.bID

    def check_temp(self) -> tuple[float, float]:
        """reads slm head & board temp

        Returns
        -------
        tuple o' floats & sets upon self
            writes temps to the class variables and returns them as floats in a tuple
        """

        self.bID = self.ffi.cast('uint8_t', self.bID)
        HeadTemp = self.ffi.new('double *')
        CBTemp = self.ffi.new('double *')
        self.slmffi.Check_Temp(self.bID, HeadTemp, CBTemp)
        return HeadTemp, CBTemp

    def display(self, image) -> None:
        """uploads uint8 phase to lcos

        Returns
        -------
        not a single thing
        """
        # import matplotlib.pyplot as plt
        # print("here we are")
        # plt.imshow(image, cmap="inferno")
        # plt.colorbar()
        # plt.show()
        # prep
        image = image.astype('uint8')
        ArraySize = self.ffi.cast('int32_t', image.shape[1] * image.shape[0])
        XPixel = self.ffi.cast('uint32_t', image.shape[1])
        YPixel = self.ffi.cast('uint32_t', image.shape[0])
        SlotNo = self.ffi.cast('uint32_t', 0)
        ArrayIn = self.ffi.new('uint8_t [{}]'.format(image.shape[1] * image.shape[0]))
        # 2D 2 1D
        image = image.flatten()
        for i, el in enumerate(ArrayIn):
            ArrayIn[i] = self.ffi.cast('uint8_t', image[i])
        bID = self.ffi.cast('uint8_t', self.bID)
        """upload, the func itself"""
        self.slmffi.Write_FMemArray(bID, ArrayIn, ArraySize, XPixel, YPixel, SlotNo)
        time.sleep(0.1)  # fixme: can this be reduced?
        # print("uploaded")
        # self.end()
        # print("loaded phuz")

    def run(self):
        self.load_phase(self.current_phase)

    def end(self):
        """ends the thread after phase upload"""
        super(SlmDisplay, self).end()

    @property
    def meshgrid_slm(self):
        """
        Calculates an x, y meshgrid using the pixel pitch and the native resolution of the SLM.
        :return: x, y meshgrid [m].
        """
        x = np.arange(-self.slm_size[0] / 2, self.slm_size[0] / 2, self.pitch)
        return np.meshgrid(x, x)


# init slm com
slm = SlmDisplay()
slm.connect()

# es el finAl
