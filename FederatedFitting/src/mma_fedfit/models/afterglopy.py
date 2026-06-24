from typing import Dict
import numpy as np
from .model import Model
import afterglowpy as grb
import astropy.units as u


class Afterglowpy(Model):
    def flux_density(self, x: np.array, nu: np.array, Z: Dict[str, float]) -> np.array:
        # canonical parameters are
        # PARAMS = ['thetaObs', 'thetaCore', 'thetaWing', 'p', 'logE0', 'logn0', 'xiN',
        #                   'logEpsilon_e',
        #                   'logEpsilon_B', 'z', 'DL']
        # so convert where necessary

        d_L = (Z['DL'] * u.Mpc).to(u.cm).value  # Turn it to cm for the fitting

        E0 = 10 ** Z['logE0']
        n0 = 10 ** Z['logn0']
        epsilon_e = 10 ** Z['logEpsilon_e']
        epsilon_B = 10 ** Z['logEpsilon_B']
        Z2 = {
            'jetType': grb.jet.Gaussian,  # Jet type
            'specType': 0,  # Basic Synchrotron Emission Spectrum
            'thetaObs': Z['thetaObs'],
            'E0': E0,
            'thetaCore': Z['thetaCore'],
            'thetaWing': Z['thetaWing'],
            'n0': n0,
            'p': Z['p'],
            'epsilon_e': epsilon_e,
            "epsilon_B": epsilon_B,
            "xi_N": 1.0,  # Fraction of electrons accelerated
            "d_L": d_L,
            'z': Z['z'],
        }
        return grb.fluxDensity(x, nu, **Z2)