from pymoo.util.misc import vectorized_cdist
import numpy as np


class IGDX:
    def __init__(self, ps):
        self.ps = ps

    def calc(self, X):
        D = vectorized_cdist(self.ps, X)
        return np.mean(np.min(D, axis=1))
