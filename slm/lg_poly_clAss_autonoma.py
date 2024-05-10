""" https://diffractio.readthedocs.io/en/latest/_modules/diffractio/scalar_sources_XY.html#Scalar_source_XY.hermite_gauss_beam """

from math import factorial

import matplotlib.pyplot as plt
import numpy as np


def laguerre_polynomial_nk(x, n=4, k=5):
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

        """

    f = factorial
    summation = np.zeros_like(x, dtype=float)
    for m in range(n + 1):
        summation = summation + (-1)**m * f(n + k) / (f(n - m) * f(k + m) *
                                                      f(m)) * x**m
    return summation


class Beamz:
    """ Class for light sources """

    def __init__(self):
        self.lgPhu_mod = []
        self.lgInt = []
        self.lg = []
        self.slm_pxlX = 1272
        self.slm_pxlY = 1024
        self.wavelength = 752

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
        axis = np.linspace(-1, 1, self.slm_pxlY) * self.slm_pxlY  # axis = np.linspace(-1, 1, axislength) * 1024
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
        k = 2 * np.pi / (self.wavelength/1000)

        # Calculate propagation
        zR = k * w0 ** 2 / 2
        w = w0 * np.sqrt(1 + z ** 2 / zR ** 2)
        if z == 0:
            R = np.inf
        else:
            R = z + zR ** 2 / z

        # Calculate amplitude
        A = A * w0 / w
        Er = laguerre_polynomial_nk(2 * Ro2 / w ** 2, order_rdl, order_ang) * np.exp(
            -Ro2 / w ** 2) * (r2 * Ro / w) ** order_ang

        # Calculate phase
        Ef = np.exp(1j * (k * Ro2 / R + order_ang * Th)) * \
            np.exp(-1j * (1 + order_rdl) * np.arctan(z / zR))

        lgComp = A * Er * Ef
        self.lg = lgComp
        self.lgPhu_mod = np.mod(np.angle(lgComp), 2 * np.pi)
        self.lgInt = np.abs(lgComp) ** 2


# cAll the clAss
bb = Beamz()

# go
bb.laguerre_beam(r0=(0, 0), A=1, order_rdl=0, order_ang=1, z=0.01, z0=0)
ziOne = bb.lgPhu_mod

plt.imshow(ziOne, cmap='magma')
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
