from typing import Dict
import numpy as np
from .model import Model
import afterglowpy as grb
import astropy.units as u


class Fake(Model):
    def flux_density(self, x: np.array, nu: np.array, Z: Dict[str, float]) -> float:
        return np.zeros(len(x))