"""
    Analytic solutions to Fickian diffusion problem
"""

import numpy as np
import matplotlib.pyplot as plt
from thermo import water_psat


def cxt1(x, t):
    l = 3.0
    D = 4.0
    psq = np.pi*np.pi
    c = 10*np.sin(np.pi*x/l)*np.exp(-psq/l*D*t) \
        + 2*np.sin(20*np.pi*x/l)*np.exp(-400*psq/l*D*t) \
        + np.sin(10*np.pi*x/l)*np.exp(-100*psq/l*D*t)
    return c


def cxt2(D, x, t, N, l, kP):
    """Analytic solution for 1D Fickian diffusion problem
    """

    c = kP - kP*x/l
    for n in range(1, N):
        c += -2*kP/(n * np.pi) * np.sin(n*np.pi*x/l) \
            * np.exp(-D*n**2*np.pi*np.pi/l/l*t)
    return c


def wvtr_vec_global(times, RH, Ta, l_mil,  D0, C0, D1, C1, t0=0.):
    D = D0*np.exp(D1/(Ta+273.15))
    C = C0*np.exp(C1/(Ta+273.15))
    return wvtr_vec(times, RH, Ta, l_mil, D, C, t0=0.)


def wvtr_vec(times, RH, Ta, l_mil, D, C, t0=0.):
    n = 20
    l = l_mil*2.54e-5
    Pa = 101.3325*1000

    activity = RH * water_psat(Ta)/Pa

    prefac = D*C*activity/l
    series = 1.0
    pisq = np.pi*np.pi
    t0 = 80
    for n in range(1, n+1):
        series += 2.*(-1)**n*np.exp(-D*n*n*pisq*(times-80)/l/l)
    series[np.where(times-t0 < 10)] = 0.0
    return prefac*series


if __name__ == "__main__":
    x = np.linspace(0, 3, 2000)
    f = cxt1(x, 0.0)
    plt.plot(x, f)
    for i in range(20):
        f2 = cxt1(x, i*1.0e-4)
        plt.plot(x, f2)
    plt.figure()
    L = 7*2.54e-5*100
    kP = 2.38e-3  # this is 'k' from curve fit

    x = np.linspace(0, L, 50)
    flux = []
    D = 6.544e-9  # Should be in cm^2/s
    l = 7*2.54e-5*100  # mil -> cm
    for i in range(20):
        f = cxt2(D, x, i*1000.0, 11, l, kP)
        plt.plot(x, f)
        df = (f[-1] - f[-2])/(x[-1] - x[-2])
        flux.append(D*df)

    plt.figure()
    plt.plot(np.abs(np.array(flux)))
    plt.show()
