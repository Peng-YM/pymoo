import numpy as np
from numpy import sqrt, zeros, pi, linspace, cos, double, vstack

from pymoo.problems.multi.mmf import MMF


class MMF2Base(MMF):
    def __init__(self, vertical_shift):
        assert vertical_shift == 1 or vertical_shift == 0.5
        self.v = vertical_shift  # shift the upper curve
        super().__init__(
            n_var=2, n_obj=2, n_constr=0, type_var=double,
            xl=[0, 0], xu=[1, 1 + self.v]
        )

    def get_segments(self, X):
        raise NotImplementedError

    def _evaluate(self, X, out, *args, **kwargs):
        F = zeros((len(X), self.n_var))
        X1 = X[:, 0]
        X2 = X[:, 1]

        J1, J2 = self.get_segments(X)

        # The two Pareto sets should not have any overlapped part and they should cover all the solutions.
        # assert np.all((J1 & J2) is False) and np.all((J1 | J2) is True)

        F[:, 0] = X1
        F[J1, 1] = 1 - sqrt(X1[J1]) + 2 * (4 * (X2[J1] - sqrt(X1[J1])) ** 2 - 2 * cos(
            (20 * (X2[J1] - sqrt(X1[J1])) * pi) / sqrt(2)) + 2)
        F[J2, 1] = 1 - sqrt(X1[J2]) + 2 * (4 * (X2[J2] - self.v - sqrt(X1[J2])) ** 2 - 2 * cos(
            (20 * (X2[J2] - self.v - sqrt(X1[J2])) * pi) / sqrt(2)) + 2)
        out["F"] = F

    def _calc_pareto_set(self, n_pareto_points=500):
        h = int(n_pareto_points / 2)

        PS1 = zeros((h, self.n_var))
        PS1[:, 1] = linspace(0, 1, h)
        PS1[:, 0] = PS1[:, 1] ** 2

        PS2 = PS1.copy()
        PS2[:, 1] = PS1[:, 1] + self.v

        return vstack((PS1, PS2))


class MMF2(MMF2Base):
    def __init__(self):
        super().__init__(vertical_shift=1)

    def get_segments(self, X):
        X1 = X[:, 0]
        X2 = X[:, 1]
        J1 = (0 <= X2) & (X2 <= 1)
        J2 = (1 < X2) & (X2 <= 2)
        # fix boundary cases
        J2 = J2 | ((X1 == 0) & (X2 == 1))
        return np.array(J1), np.array(J2)


class MMF3(MMF2Base):
    def __init__(self):
        super().__init__(vertical_shift=0.5)

    def get_segments(self, X):
        X1 = X[:, 0]
        X2 = X[:, 1]
        J1 = ((X2 >= 0) & (X2 < 0.5)) | ((X2 > 0.5) & (X2 <= 1) & (X1 > 0.25) & (X1 <= 1))
        J2 = ((X2 > 1) & (X2 <= 1.5)) | ((X2 >= 0.5) & (X2 < 1) & (X1 >= 0) & (X1 < 0.25))
        return np.array(J1), np.array(J2)
