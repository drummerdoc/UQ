import numpy as np


def water_psat(Ta):
    """
    Return saturation pressure of water at given temperature in Pa, T in C
    Uses data from G. van Wylen, R. Sonntag and C. Borgnakke, Fundamentals of
    Classical Thermodynamics, 4th ed. John Wiley & Sons, Inc., New York. 1993.
    ISBN 0-471-59395-8.
    """
    Tsat = np.array([0.01, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                     55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    Psat = np.array([0.6113, 0.8721, 1.2276, 1.7051, 2.3385, 3.1691,
                     4.2461, 5.6280, 7.3837, 9.5934, 12.350, 15.758,
                     19.941, 25.033, 31.188, 38.578, 47.390, 57.834,
                     70.139, 84.554, 101.325])
    return np.interp(Ta, Tsat, Psat*1000)
