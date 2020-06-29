from pymoo.performance_indicator.igdx import IGDX
import numpy as np


class PSP:
    def __init__(self, ps, normalize=False):
        self.ps = ps
        self.normalize = normalize

    def calc(self, X):
        v_max, v_min = np.max(self.ps, axis=0), np.min(self.ps, axis=0)
        x_max, x_min = np.max(X, axis=0), np.min(X, axis=0)

        CR = 0
        n_var = X.shape[1]

        for i in range(n_var):
            if v_min[i] == v_max[i]:
                CR += 1
            elif x_min[i] >= v_max[i] or x_max[i] <= v_min[i]:
                CR += 0
            else:
                CR += ((min(v_max[i], x_max[i]) - max(v_min[i], x_min[i])) / (v_max[i] - v_min[i])) ** 2

        CR = CR ** (1 / 2 / n_var)

        return CR / IGDX(self.ps).calc(X)
