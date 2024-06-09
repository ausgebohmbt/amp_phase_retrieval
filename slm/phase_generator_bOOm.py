import numpy as np
from random import randint, random
import os
import glob
from PIL import Image
from slm.helpers import unimOD, normalize, center_overlay, center_crop, tiler
import matplotlib.pyplot as plt
from math import factorial


class PhaseGen:
    """cre'tes and treats the phase functions for the slm"""
    def __init__(self):
        self.f = 624424  # *1e-3
        self.pitch = 12.5  # *1e-6
        self.lamda = 752  # *1e-9
        self.slmX = 1272
        self.slmY = 1024
        self.D = self.slmY
        self.diviX = 5
        self.diviY = 0
        self.width_X = 1
        self.width_Y = 1
        self.canVas = np.zeros((self.slmY, self.slmX))
        self.crxn_pattie = np.zeros((self.slmY, self.slmX))
        self.fou = self.canVas
        self.grat = self.canVas
        self.phase = self.canVas
        self.amp = self.canVas
        self.phase_prev = self.canVas
        self.phase_init = self.canVas
        self.amp_prev = self.canVas
        self.state_grat = False
        self.state_lens = False
        self.state_phase = False
        self.state_amp = False
        self.state_corr = False
        self.state_lg = True
        self.state_cor_phase = False
        self.whichphuzzez = {"grating": self.state_grat, "lens": self.state_lens,
                             "phase": self.state_phase, "amplitude": self.state_amp, "corr_patt": self.state_corr,
                             "lg_pol": self.state_lg, "corr_phase": self.state_cor_phase}
        self.final_phuz = self.canVas
        self.modDepth = 220
        self.xprmental_prmtrs = {"spin_conf": True, "amp_conf": True,
                                 "side_fill": False}
        self.spin_array_sz = (800, 800)
        self.lgPhu_mod = self.canVas
        self.lgInt = self.canVas
        self.lg = self.canVas
        self.crxn_phuz_boom = self.canVas

        # creAte 1020x1020 rundum array for use as amp
        cnVs = np.zeros((1020, 1020))
        randy = [random() for r in range(41616)]
        pxl = [[0. for i in range(5)] for j in range(5)]
        pxl_sq = np.asarray(pxl).reshape((5, 5))

        rr = 0
        for i in range(204):
            for j in range(204):
                pxl_sq[:] = randy[rr]
                cnVs[i * 5:i * 5 + 5, j * 5:j * 5 + 5] = pxl_sq
                rr += 1
        self.random = cnVs

        # creAte 1020x1020 checkerboard for use as AFM phase
        #  ega 5x5
        che_ck_err = np.ones((1020, 1020))
        pxle = [[1. for i in range(5)] for j in range(5)]
        pxl_err = np.asarray(pxle).reshape((5, 5))

        for m in range(204):
            for n in range(204):
                if m % 2 == 0 and n % 2 == 0:
                    pxl_err[:] = 0
                    che_ck_err[n * 5:n * 5 + 5, m * 5:m * 5 + 5] = pxl_err
                elif m % 2 != 0 and n % 2 != 0:
                    pxl_err[:] = 0
                    che_ck_err[n * 5:n * 5 + 5, m * 5:m * 5 + 5] = pxl_err
        self.checker_bo = che_ck_err

    def berg_phuz(self):
        """the phase, FM or AM"""
        """current implementation uses a 5x5 checkerboard phuz created during init"""
        if self.xprmental_prmtrs["spin_conf"]:
            self.phase = center_overlay(self.slmY, self.slmY, self.checker_bo)
        else:
            self.phase = center_overlay(self.slmY, self.slmY, np.ones((1020, 1020)))
        self._make_full_slm_array()

    def random_flip(self):
        """flips a random 5x5 'egapixel"""
        randy_aye = randint(0, 204)
        randy_jay = randint(0, 204)
        randy_Andy = self.phase
        if randy_Andy[randy_aye*5:randy_aye*5 + 5, randy_jay*5:randy_jay*5 + 5] == 0:
            randy_Andy[randy_aye * 5:randy_aye * 5 + 5, randy_jay * 5:randy_jay * 5 + 5] = 1
        else:
            randy_Andy[randy_aye * 5:randy_aye * 5 + 5, randy_jay * 5:randy_jay * 5 + 5] = 0
        self.phase = randy_Andy

    def berg_amp(self):
        """the 'amplitude', random or filler when the -fill the cup option is on-"""
        """current implementation uses a 1020x1020 random 0 2 1 5x5 megaPixel array created during init"""
        if self.xprmental_prmtrs["amp_conf"]:
            self.amp = center_overlay(self.slmY, self.slmY, self.random)
        else:
            self.amp = center_overlay(self.slmY, self.slmY, 1-self.checker_bo)  # cup filler
        self._make_full_slm_array()

    def correction_patt(self):
        """load bmp and add to phase"""
        current_path = os.getcwd()
        print(current_path)
        if "tests" in current_path:
            current_path = current_path.replace("\\tests", "")
        img_nom = glob.glob(current_path + "\\slm\\corr_patties\\CAL_LSH0803174_750nm.bmp")
        print(img_nom)
        if os.path.exists(img_nom[0]):  # check if pattie exists
            with Image.open(img_nom[0]) as img:
                im = np.array(img).astype(np.float32)
            self.crxn_pattie = normalize(np.asarray(im, dtype=np.uint16))  # lOaD as aRRay
            self._make_full_slm_array()
        else:
            print('correction pattern bmp does not exist')

    def fourier_lens(self):
        """Same equation inthesis' of Jesacher &  ref [1] of Padgett,
            only difference between them is a minus
            sign in the latter, we ll keep it as he is the ,aster on the matter,,,.
            Also the Goodman expression which coicides with that of Gaunt is verified to give
            the same result to e-23 accuracy, though ca check for thine selv
            Somethin that need be considere is what going on with the units is?
            I ll start by using micrometers [m]

            [1] Interactive approach to optical tweezers control, Leach J. et al Applied Optics 2006"""
        um = 1000000
        # um = 1
        diameter = (self.D * (self.pitch/um))
        y = np.linspace(-diameter/2, diameter/2, self.D)
        x = y
        phaseX = x**2
        phaseY = y**2

        xx, yy = np.meshgrid(phaseX, phaseY)

        # # these work
        lenz = - (np.pi * (xx + yy))
        lenz = np.mod(lenz, ((self.lamda/(um*1000)) * (self.f / 1000)))  # /1000 to convert to μm

        tileDlgPhu = tiler(lenz, self.slmY, self.slmY)
        cntrdHolo = center_crop(tileDlgPhu, self.slmY, self.slmY)
        self.fou = normalize(cntrdHolo)
        self._make_full_slm_array()

    """"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    """ phase functions as imported from ising """
    """"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    def linear_grating(self):
        # fixme: the vertical grating whe used with the horizontal one destroys everything, take it out
        # you gotta keep em separated
        # todo: comment this proper-ly
        """ex-grati_x_wide(div, sz_x, sz_y, width)
        Generate a linear grating array.

        The pattern repeats every ``div`` pixels and spans the entire specified
        ``shape`` of the output array.

        Parameters
        ----------
        shape : tuple[int, int]
            The shape of the grating pattern to be generated. It is specified as (nrows, ncols),
            where nrows is the number of rows and ncols is the number of columns in the pattern.
        div : int [the initial function is using float to express frequency]
            It sets the number of steps in pixels over which the phase repeats.
            We typically use a 5 step grating so that we get an integer result when converting to uint8.
            If positive, the grating rises from left to right. 
            If negative you get a prompt and the same result as if it where positive. 
            If zero the function returns a uniform array of zeros.
        width: here it is hardcoded to 1. You can use it to get a wider step for each pixel value

        Examples
        --------
        # >>> grating = linear_grating(shape=(100, 200), div=2)
        # >>> grating.shape
        (100, 200)

        # >>> grating = linear_grating(shape=(2, 4), div=2)
        # >>> grating
        array([[0.        , 1, 0.        , 1],
               [0.        , 1, 0.        , 1]])"""
        width = 1
        # fixme: rm redundancy
        sz_y = self.slmY
        sz_x = self.slmX
        div_x = self.diviX
        if sz_x >= sz_y:
            sz = sz_x
        elif sz_y > sz_x:
            sz = sz_y
        else:
            sz = sz_x
        if div_x != 0:
            x = np.linspace(0, sz, sz)
            x = np.mod(np.floor(x), div_x) / div_x
            # repeat elements K times
            res = []
            for i in x:
                for ele in range(width):
                    res.append(i)
            grr = np.tile(res, (len(res), 1))
        else:
            grr = np.zeros((sz_y, sz_x))
        if div_x < 0:
            print("grating divisions number is negative, only positive values are supported\r\n")

        grat = grr[:self.slmY, :self.slmX]
        # grat = np.fliplr(agrr_x)
        self.grat = grat
        self._make_full_slm_array()

    def _make_full_slm_array(self):

        if self.whichphuzzez["grating"]:
            gra = self.grat
        else:
            gra = self.canVas
        if self.whichphuzzez["lens"]:
            le = self.fou
        else:
            le = self.canVas
        if self.whichphuzzez["phase"]:
            phu = self.phase
        else:
            phu = self.canVas
        if self.whichphuzzez["amplitude"]:
            am = self.amp
        else:
            am = self.canVas
        if self.whichphuzzez["corr_patt"]:
            crxn_pat = self.crxn_pattie
        else:
            crxn_pa = self.canVas
            crxn_pat = center_overlay(self.slmX, self.slmY, crxn_pa)
        if self.whichphuzzez["lg_pol"]:
            lg = center_overlay(self.slmX, self.slmY, phagen.lgPhu_mod)
        else:
            lg = center_overlay(self.slmX, self.slmY, self.canVas)
        if self.whichphuzzez["corr_phase"]:
            core_phuz = self.crxn_phuz_boom
            core_phuz = center_overlay(self.slmX, self.slmY, self.crxn_phuz_boom)
        else:
            core_phuz = center_overlay(self.slmX, self.slmY, self.canVas)

        # combined = np.add(np.add(gra, le), np.add(phu, am))
        # mOD = unimOD(combined)
        # pha_ce = center_overlay(self.slmX, self.slmY, mOD)
        # print(self.whichphuzzez)


        # ------------ lg case --------------------
        comb = unimOD(np.add(crxn_pat, gra))
        combo = unimOD(np.add(comb, lg))
        comboom = np.add(combo, core_phuz)
        # -----------------------------------------

        phuz = unimOD(comboom) * self.modDepth
        self.final_phuz = phuz.astype('uint8')

    def spin_to_phase(self, spin: np.ndarray, theta: float) -> np.ndarray:
        # fixme: check this thoroughly, I think that by introducing the mod-depth from the slm [or even brettah include
        #  this functio to the class or a separate one for the phases] and converting the angle variable
        #  'theta' accordingly it will work
        r"""Maps spins to phases.

        The mapping is

        .. math::
            \phi = \frac{s + 1}{2} \pi + \theta.

        Examples
        --------
        #>>> spins = np.array([1, -1])
        #>>> spin_to_phase(spins, theta=0)
        array([3.14159265, 0.        ])
        #>>> spin_to_phase(spins, theta=np.pi / 2)
        array([4.71238898, 1.57079633])
        """
        # modDepth = 100  # fixme fixmeeeeeeeee
        # return np.pi * ((spin + 1) / 2) + np.array(theta)
        if theta != 0:
            print("theta ain't zero mate")
        # return modDepth * ((spin + 1) / 2) + np.array(theta)
        return ((spin + 1) / 2) + np.array(theta)

    def upscale(self, array: np.ndarray, factor: int):
        # fixme: replace this
        """Upscales a 2D numpy array by a given factor along both axes using repetition.

        Examples
        --------
        #>>> x = numpy.array([[1, 2],
                             [3, 4]])
        #>>> upscale(x, 2)
        array([[1, 1, 2, 2],
               [1, 1, 2, 2],
               [3, 3, 4, 4],
               [3, 3, 4, 4]])
        """
        if factor < 0:
            raise ValueError("The upscale factor must be non negative.")

        if array.ndim != 2:
            raise ValueError("The input array must be a 2D numpy array.")
        upscaled = array.repeat(factor, axis=0).repeat(factor, axis=1)
        self.spin_array_sz = upscaled.shape
        self.phase_prev = self.phase
        self.phase = center_overlay(self.slmY, self.slmY, upscaled) / 2
        # todo: this feels wrong... maybe use the established methOD of "self.states"?
        self.whichphuzzez = {"grating": True, "lens": False, "phase": True, "amplitude": False,
                             "corr_patt": self.state_corr}
        # self.whichphuzzez = {"grating": True, "lens": False, "phase": True, "amplitude": False, "corr_patt": True}
        self._make_full_slm_array()

    def checkerboard(self, shape: tuple[int, ...]) -> np.ndarray:
        # fixme: replace this
        """Creates a checkerboard pattern numpy array of 0 and 1, of given shape.

        Examples
        --------
        #>>> checkerboard((3, 3))
        array([[0, 1, 0],
               [1, 0, 1],
               [0, 1, 0]])
        """
        return np.indices(shape).sum(axis=0) % 2

    def laguerre_polynomial_nk(self, x, n=4, k=5):
        """Auxiliar laguerre polinomial of orders n and k
            function y = LaguerreGen(varargin)
            LaguerreGen calculates the utilsized Laguerre polynomial L{n, alpha}
            This function computes the utilsized Laguerre polynomial L{n,alpha}.
            If no alpha is supplied, alpha is set to zero and this function
            calculates the "normal" Laguerre polynomial.


            Parameters:
            - n = nonnegative integer as degree level
            - alpha >= -1 real number (input is optional)

            The output is formated as a polynomial vector of degree (n+1)
            corresponding to MatLab norms (that is the highest coefficient
            is the first element).

            Example:
            - polyval(LaguerreGen(n, alpha), x) evaluates L{n, alpha}(x)
            - roots(LaguerreGen(n, alpha)) calculates roots of L{n, alpha}

            Calculation is done recursively using matrix operations for very fast
            execution time.

            Author: Matthias.Trampisch@rub.de
            Date: 16.08.2007
            Version 1.2

            References:
                Szeg: "Orthogonal Polynomials" 1958, formula (5.1.10)
                https://diffractio.readthedocs.io/en/latest/_modules/diffractio/scalar_sources_XY.html#Scalar_source_XY.hermite_gauss_beam
            """

        f = factorial
        summation = np.zeros_like(x, dtype=float)
        for m in range(n + 1):
            summation = summation + (-1)**m * f(n + k) / (f(n - m) * f(k + m) *
                                                          f(m)) * x**m
        return summation

    def laguerre_beam(self, r0, A, order_rdl, order_ang, z, z0):
        """Laguerre beam.

        Parameters:
            A (float): amplitude of the Hermite Gauss beam.
            r0 (float, float): (x,y) position of the beam center.
            w0 (float): Gaussian waist.
            order_rdl (int): radial order.
            order_ang (int): angular order.
            z (float): Propagation distance.
            z0 (float): Beam waist position.

        Example:
            laguerre_beam(A=1, r0=(0 * um, 0 * um),  w0=1 * um,  p=0, l=0,  z=0)
        """
        # Prepare space
        axis = np.linspace(-1, 1, self.slmY) * self.slmY  # axis = np.linspace(-1, 1, axislength) * 1024
        X, Y = np.meshgrid(axis, axis)
        w0 = int(len(Y) / 2)
        X = X - r0[0]
        Y = Y - r0[1]
        Ro2 = X ** 2 + Y ** 2
        Ro = np.sqrt(Ro2)
        Th = np.arctan2(Y, X)

        # Parameters
        r2 = np.sqrt(2)
        z = z - z0
        k = 2 * np.pi / (self.lamda/1000)

        # Calculate propagation
        zR = k * w0 ** 2 / 2
        w = w0 * np.sqrt(1 + z ** 2 / zR ** 2)
        if z == 0:
            R = np.inf
        else:
            R = z + zR ** 2 / z

        # Calculate amplitude
        A = A * w0 / w
        Er = self.laguerre_polynomial_nk(2 * Ro2 / w ** 2, order_rdl, order_ang) * np.exp(
            -Ro2 / w ** 2) * (r2 * Ro / w) ** order_ang

        # Calculate phase
        Ef = np.exp(1j * (k * Ro2 / R + order_ang * Th)) * \
            np.exp(-1j * (1 + order_rdl) * np.arctan(z / zR))

        lgComp = A * Er * Ef
        self.lg = lgComp
        self.lgPhu_mod = normalize( np.mod(np.angle(lgComp), 2 * np.pi))
        self.lgInt = np.abs(lgComp) ** 2


phagen = PhaseGen()
phagen.correction_patt()
phagen.laguerre_beam(r0=(0, 0), A=1, order_rdl=0, order_ang=1, z=0.01, z0=0)

# es el finAl
