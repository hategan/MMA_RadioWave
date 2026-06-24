from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class Model(ABC):
    @abstractmethod
    def flux_density(self, x: np.array, nu: np.array, Z: Dict[str, float]) -> float:
        pass