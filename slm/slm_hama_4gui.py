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
from PyQt5 import QtCore


class SLM_clash(QtCore.QThread):
    """Base class for all Spatial Light Modulators."""
    __metaclass__ = abc.ABCMeta
    # def signals to trigger main
    phuz_up = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(SLM_clash, self).__init__(parent)

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
        self.current_phase = np.zeros((self.slmY, self.slmX))

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

    def load_phase(self, image) -> None:
        """uploads uint8 phase to lcos

        Returns
        -------
        not a single thing
        """
        # plt.imshow(image, cmap="inferno")
        # plt.colorbar()
        # plt.show()
        # prep
        image = image.astype('uint8')
        ArraySize = self.ffi.new('int32_t*')
        ArraySize = self.ffi.cast('int32_t', image.shape[1] * image.shape[0])
        XPixel = self.ffi.new('uint32_t*')
        XPixel = self.ffi.cast('uint32_t', image.shape[1])
        YPixel = self.ffi.new('uint32_t*')
        YPixel = self.ffi.cast('uint32_t', image.shape[0])
        SlotNo = self.ffi.new('uint32_t*')
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
        self.phuz_up.emit()
        self.end()
        # print("loaded phuz")

    def run(self):
        self.load_phase(self.current_phase)

    def end(self):
        """ends the thread after phase upload"""
        super(SlmDisplay, self).end()


# init slm com
slm = SlmDisplay()
slm.connect()


#     """"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
#     """ phase functions section"""
#     """"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
#
#     # FIXME: am flirting with the idea to have those all in a class as in the gui version
#
#     def correction_patt(self):
#         """load bmp and add to phase"""
#         current_path = os.getcwd() + r"\slm"
#         img_nom = glob.glob(current_path + "\\corr_patties\\CAL_LSH0803174_750nm.bmp")
#         if os.path.exists(img_nom[0]):  # check if pattie exists
#             with Image.open(img_nom[0]) as img:
#                 im = np.array(img).astype(np.float32)
#             self.crxn_pattie = normalize(np.asarray(im, dtype=np.uint16))  # lOaD as aRRay
#             self._make_full_slm_array()
#         else:
#             print('correction pattern bmp does not exist')
#
#     def linear_grating(self, shape: tuple[int, int], div: int) -> np.ndarray:
#         """ex-grati_x_wide(div, sz_x, sz_y, width)"""
#         """Generate a linear grating array.
#
#         The pattern repeats every ``div`` pixels and spans the entire specified
#         ``shape`` of the output array.
#
#         Parameters
#         ----------
#         shape : tuple[int, int]
#             The shape of the grating pattern to be generated. It is specified as (nrows, ncols),
#             where nrows is the number of rows and ncols is the number of columns in the pattern.
#         div : int [the initial function is using float to express frequency]
#             It sets the number of steps in pixels over which the phase repeats.
#             We typically use a 5 step grating so that we get an integer result when converting to uint8.
#             If positive, the grating rises from left to right.
#             If negative you get a prompt and the same result as if it where positive.
#             If zero the function returns a uniform array of zeros.
#         width: here it is hardcoded to 1. You can use it to get a wider step for each pixel value
#
#         Examples
#         --------
#         >>> grating = linear_grating(shape=(100, 200), div=2)
#         >>> grating.shape
#         (100, 200)
#
#         >>> grating = linear_grating(shape=(2, 4), div=2)
#         >>> grating
#         array([[0.        , 1, 0.        , 1],
#                [0.        , 1, 0.        , 1]])"""
#         width = 1
#         sz_y, sz_x = shape
#         if sz_x >= sz_y:
#             sz = sz_x
#         elif sz_y > sz_x:
#             sz = sz_y
#         else:
#             sz = sz_x
#         if div != 0:
#             x = np.linspace(0, sz, sz)
#             x = np.mod(np.floor(x), div) / div
#             # repeat elements K times
#             res = []
#             for i in x:
#                 for ele in range(width):
#                     res.append(i)
#             grr = np.tile(res, (len(res), 1))
#         else:
#             grr = np.zeros((sz_y, sz_x))
#         if div < 0:
#             print("grating divisions number is negative, only positive values are supported\r\n")
#         grat = grr[:800, :800]
#         self.grat = center_overlay(self.slmY, self.slmY, grat)
#
#         # self.grat = grr[:self.slmY, :self.slmY]
#         self._make_full_slm_array()
#
#     # def checkerboard(self, shape: tuple[int, ...]) -> np.ndarray:
#     #     """Creates a 1020x1020 checkerboard pattern numpy array of 0 and 1 of fixed 5x5 'megapixel' arrangement"""
#     #     # TODO: fix meeeeeeeeeeeeee
#     #     sz_y, sz_x = shape
#     #     print("you wanted a " + str(sz_y) + "x" + str(sz_x) + "array, "
#     #                                                           "but a 1020x1020 divided in 5x5 megapixels you will get")
#     #     che_ck_err = np.ones((1020, 1020))
#     #     pxle = [[1. for i in range(5)] for j in range(5)]
#     #     pxl_err = np.asarray(pxle).reshape((5, 5))
#     #
#     #     for m in range(204):
#     #         for n in range(204):
#     #             if m % 2 == 0 and n % 2 == 0:
#     #                 pxl_err[:] = 0
#     #                 che_ck_err[n * 5:n * 5 + 5, m * 5:m * 5 + 5] = pxl_err
#     #             elif m % 2 != 0 and n % 2 != 0:
#     #                 pxl_err[:] = 0
#     #                 che_ck_err[n * 5:n * 5 + 5, m * 5:m * 5 + 5] = pxl_err
#     #     self.phase = center_overlay(self.slmY, self.slmY, che_ck_err)
#     #     self._make_full_slm_array()
#     #
#     # def upscale(self, array: np.ndarray, factor: int) -> np.ndarray:
#     #     # TODO: check it, looks robust-o
#     #     """Upscales a 2D numpy array by a given factor along both axes using repetition.
#     #
#     #     Examples
#     #     --------
#     #     >>> x = numpy.array([[1, 2],
#     #                          [3, 4]])
#     #     >>> upscale(x, 2)
#     #     array([[1, 1, 2, 2],
#     #            [1, 1, 2, 2],
#     #            [3, 3, 4, 4],
#     #            [3, 3, 4, 4]])
#     #     """
#     #     if factor < 0:
#     #         raise ValueError("The upscale factor must be non negative.")
#     #
#     #     if array.ndim != 2:
#     #         raise ValueError("The input array must be a 2D numpy array.")
#     #     return array.repeat(factor, axis=0).repeat(factor, axis=1)
#     #
#     # def spin_to_phase(self, spin: np.ndarray, theta: float) -> np.ndarray:
#     #     # TODO: this here be the main thing we change need must, aka the shit
#     #     r"""Maps spins to phases.
#     #
#     #     The mapping is
#     #
#     #     .. math::
#     #         \phi = \frac{s + 1}{2} \pi + \theta.
#     #
#     #     Examples
#     #     --------
#     #     >>> spins = np.array([1, -1])
#     #     >>> spin_to_phase(spins, theta=0)
#     #     array([3.14159265, 0.        ])
#     #     >>> spin_to_phase(spins, theta=np.pi / 2)
#     #     array([4.71238898, 1.57079633])
#     #     """
#     #     return np.pi * ((spin + 1) / 2) + np.array(theta)
#
#     def _make_full_slm_array(self):
#
#         if self.whichphuzzez["grating"]:
#             gra = self.grat
#         else:
#             gra = self.canVas
#         if self.whichphuzzez["lens"]:
#             le = self.fou
#         else:
#             le = self.canVas
#         if self.whichphuzzez["phase"]:
#             phu = self.phase
#         else:
#             phu = self.canVas
#         if self.whichphuzzez["amplitude"]:
#             am = self.amp
#         else:
#             am = self.canVas
#
#         combined = np.add(np.add(gra, le), np.add(phu, am))
#         mOD = unimOD(combined)
#         pha_ce = center_overlay(self.slmX, self.slmY, mOD)
#         combo = np.add(self.crxn_pattie, pha_ce)
#         phuz = unimOD(combo)*self.modDepth
#         self.final_phuz = phuz.astype('uint8')
#
#         # plt.subplot(131), plt.imshow(gra, cmap="inferno")
#         # plt.title("grating")
#         # plt.colorbar(fraction=0.046, pad=0.04)
#         # plt.subplot(132), plt.imshow(phu, cmap="inferno")
#         # plt.colorbar(fraction=0.046, pad=0.04)
#         # plt.title("phase")
#         # plt.subplot(133), plt.imshow(self.final_phuz, cmap="inferno")
#         # plt.colorbar(fraction=0.046, pad=0.04)
#         # plt.title("final 4 upload")
#         # plt.show()
#
#     def spin_to_phase(self, spin: np.ndarray, theta: float) -> np.ndarray:
#         # fixme: check this thoroughly, I think that by introducing the mod-depth from the slm [or even brettah include
#         #  this functio to the class or a separate one for the phases] and converting the angle variable 'theta' accordingly
#         #  it will work
#         r"""Maps spins to phases.
#
#         The mapping is
#
#         .. math::
#             \phi = \frac{s + 1}{2} \pi + \theta.
#
#         Examples
#         --------
#         >>> spins = np.array([1, -1])
#         >>> spin_to_phase(spins, theta=0)
#         array([3.14159265, 0.        ])
#         >>> spin_to_phase(spins, theta=np.pi / 2)
#         array([4.71238898, 1.57079633])
#         """
#         # modDepth = 100  # fixme fixmeeeeeeeee
#         # return np.pi * ((spin + 1) / 2) + np.array(theta)
#         if theta != 0:
#             print("theta ain't zero mate")
#         # return modDepth * ((spin + 1) / 2) + np.array(theta)
#         return ((spin + 1) / 2) + np.array(theta)
#
#     def upscale(self, array: np.ndarray, factor: int) -> np.ndarray:
#         # fixme: replace this
#         """Upscales a 2D numpy array by a given factor along both axes using repetition.
#
#         Examples
#         --------
#         >>> x = numpy.array([[1, 2],
#                              [3, 4]])
#         >>> upscale(x, 2)
#         array([[1, 1, 2, 2],
#                [1, 1, 2, 2],
#                [3, 3, 4, 4],
#                [3, 3, 4, 4]])
#         """
#         if factor < 0:
#             raise ValueError("The upscale factor must be non negative.")
#
#         if array.ndim != 2:
#             raise ValueError("The input array must be a 2D numpy array.")
#         # return array.repeat(factor, axis=0).repeat(factor, axis=1)
#         upscaled = array.repeat(factor, axis=0).repeat(factor, axis=1)
#         self.phase = center_overlay(self.slmY, self.slmY, upscaled)/2
#
#         # plt.subplot(121), plt.imshow(array, cmap="inferno")
#         # plt.title("spins_ph 4 upscale")
#         # plt.colorbar(fraction=0.046, pad=0.04)
#         # plt.subplot(122), plt.imshow(upscaled, cmap="inferno")
#         # plt.colorbar(fraction=0.046, pad=0.04)
#         # plt.title("upscaled")
#         # plt.show()
#         self.whichphuzzez = {"grating": True, "lens": False, "phase": True, "amplitude": False, "corr_patt": True}
#         self._make_full_slm_array()
#
#
# #
# # def spin_to_phase(spin: np.ndarray, theta: float) -> np.ndarray:
# #     # fixme: check this thoroughly, I think that by introducing the mod-depth from the slm [or even brettah include
# #     #  this functio to the class or a separate one for the phases] and converting the angle variable 'theta' accordingly
# #     #  it will work
# #     r"""Maps spins to phases.
# #
# #     The mapping is
# #
# #     .. math::
# #         \phi = \frac{s + 1}{2} \pi + \theta.
# #
# #     Examples
# #     --------
# #     >>> spins = np.array([1, -1])
# #     >>> spin_to_phase(spins, theta=0)
# #     array([3.14159265, 0.        ])
# #     >>> spin_to_phase(spins, theta=np.pi / 2)
# #     array([4.71238898, 1.57079633])
# #     """
# #     modDepth = 100  # fixme fixmeeeeeeeee
# #     # return np.pi * ((spin + 1) / 2) + np.array(theta)
# #     if theta != 0:
# #         print("theta ain't zero mate")
# #     return modDepth * ((spin + 1) / 2) + np.array(theta)
# #
# #
# # def upscale(array: np.ndarray, factor: int) -> np.ndarray:
# #     # fixme: replace this
# #     """Upscales a 2D numpy array by a given factor along both axes using repetition.
# #
# #     Examples
# #     --------
# #     >>> x = numpy.array([[1, 2],
# #                          [3, 4]])
# #     >>> upscale(x, 2)
# #     array([[1, 1, 2, 2],
# #            [1, 1, 2, 2],
# #            [3, 3, 4, 4],
# #            [3, 3, 4, 4]])
# #     """
# #     if factor < 0:
# #         raise ValueError("The upscale factor must be non negative.")
# #
# #     if array.ndim != 2:
# #         raise ValueError("The input array must be a 2D numpy array.")
# #     return array.repeat(factor, axis=0).repeat(factor, axis=1)
#
#
# def checkerboard(shape: tuple[int, ...]) -> np.ndarray:
#     # fixme: replace this
#     """Creates a checkerboard pattern numpy array of 0 and 1, of given shape.
#
#     Examples
#     --------
#     >>> checkerboard((3, 3))
#     array([[0, 1, 0],
#            [1, 0, 1],
#            [0, 1, 0]])
#     """
#     return np.indices(shape).sum(axis=0) % 2

# es el finAl
