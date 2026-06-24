from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class Model(ABC):
    """A base class for flux models."""
    @abstractmethod
    def flux_density(self, x: np.array, nu: np.array, Z: Dict[str, float]) -> np.array:
        """
        Calculates the flux density.

        The flux density is calculated for the given times (``x``),
        frequencies (``nu``), and model parameters (``Z``). The times and
        frequencies are 1d arrays of the same shape. The model parameters
        are specified in a standardized way and concrete implementations of
        this class must convert to specific model parameters accordingly.
        The keys in ``Z`` are:

            * ``logE0``
            * ``logn0``
            * ``logEpsilon_e``
            * ``logEpsilon_B``
            * ``thetaObs``
            * ``thetaCore``
            * ``thetaWing``
            * ``p``
            * ``d_L`` (in Mpc)
            * ``z``

        :param x: A np.array with the observation times, in seconds.
        :param nu: A np.array with the observation frequencies, in Hz.
        :param Z: A dictionary with model parameters, as described above.
        :return: An array with flux densities at the given times and frequencies.
        """
        pass