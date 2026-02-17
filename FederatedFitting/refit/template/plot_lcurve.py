import numpy as np
import matplotlib.pyplot as plt
import fiesta_wrapper as grb1
import afterglowpy as grb2
from afterglowpy import jet

cm_per_pc = 3.086e18

Z = {
    # the jetType and specType match what Fiesta was trained on; they are otherwise
    # ignored by Fiesta
    'jetType': jet.Gaussian, 
    'specType': jet.SimpleSpec,
    'thetaObs': 0.44, 
    #'thetaObs': 0.52,
    'E0': 10 ** 53.99, 
    #'E0': 10 ** 52.92,
    'thetaCore': 0.088, 
    'thetaWing': 0.55, 
    'n0': 10 ** -3.89, 
    #'n0': 10 ** -1.96,
    'p': 2.139, 
    'epsilon_e': 0.01, 
    'epsilon_B': 0.0002, 
    'xi_N': 1.0, 
    #'d_L': 25.06 * cm_per_pc, 
    'd_L': 10 * cm_per_pc,
    #'d_L': 32.42 * cm_per_pc,
    #'z': 0.0098
    'z': 0.0
}

times = np.array([24 * 3600 * np.pow(10, x / 20.0) for x in range(20, 60)])
freqs = np.array([1.5e9 for x in range(20, 60)])

fluxes1 = grb1.fluxDensity(times, freqs, **Z)
fluxes2 = grb2.fluxDensity(times, freqs, **Z)

fig, ax = plt.subplots()

ax.loglog(times, fluxes1)
ax.loglog(times, fluxes2)

plt.show()