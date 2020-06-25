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
        if n_select >= len(self.pop):
            return np.full(len(self.pop), True)

        selected = self._do(n_select, **kwargs)
        return np.where(selected)[0]

    @abstractmethod
    def _do(self, n_select, **kwargs):
        pass
