from abc import abstractmethod

import numpy as np


def normalize(V):
    V_max = np.max(V, axis=0)
    V_min = np.min(V, axis=0)
    return (V - V_min) / (V_max - V_min)


class SubsetSelection:
    def __init__(self, population):
        self.pop = population.copy()  # copy to avoid side effects

    def do(self, n_select, **kwargs):
        return self._do(n_select, **kwargs)

    @abstractmethod
    def _do(self, n_select, **kwargs):
        pass
