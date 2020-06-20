from numpy import abs, sqrt, zeros, pi, sin, linspace, e, double

from pymoo.problems.multi.mmf import MMF


class MMF1(MMF):
    def __init__(self):
        super().__init__(
            n_var=2, n_obj=2, n_constr=0, type_var=double,
            xl=[1, -1], xu=[3, 1]
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = zeros((len(X), self.n_var))
        F[:, 0] = abs(X[:, 0] - 2)
        F[:, 1] = 1 - sqrt(abs(X[:, 0] - 2)) + 2 * (X[:, 1] - sin(6 * pi * abs(X[:, 0] - 2) + pi)) ** 2
        out["F"] = F

    def _calc_pareto_set(self, n_pareto_points=500):
        PS = zeros((n_pareto_points, self.n_var))
        PS[:, 0] = linspace(1, 3, n_pareto_points)
        PS[:, 1] = sin(6 * pi * abs(PS[:, 0] - 2) + pi)
        return PS


class MMF1z(MMF):
    """
    The MMF1_z test problem.

    Parameters
    ----------
    k: k > 0, k controls the deformation degree of the global PS in x_1 in [-1, 2).
    """
    def __init__(self, k=3):
        assert (k > 0), "Invalid parameter k!"
        self.k = k
        super().__init__(
            n_var=2, n_obj=2, n_constr=0, type_var=double,
            xl=[1, -1], xu=[3, 1]
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = zeros((len(X), self.n_var))
        X1 = X[:, 0]
        X2 = X[:, 1]

        # segments
        J1 = (1 <= X1) & (X1 < 2)
        J2 = (2 <= X1) & (X1 <= 3)

        F[:, 0] = abs(X1 - 2)
        F[J1, 1] = 1 - sqrt(abs(X1[J1] - 2)) + 2 * (X2[J1] - sin(2 * self.k * pi * abs(X1[J1] - 2) + pi)) ** 2
        F[J2, 1] = 1 - sqrt(abs(X1[J2] - 2)) + 2 * (X2[J2] - sin(2 * pi * abs(X1[J2] - 2) + pi)) ** 2
        out["F"] = F

    def _calc_pareto_set(self, n_pareto_points=500):
        PS = zeros((n_pareto_points, self.n_var))
        X1 = linspace(1, 3, n_pareto_points)

        # segments
        J1 = (1 <= X1) & (X1 < 2)
        J2 = (2 <= X1) & (X1 <= 3)

        PS[:, 0] = X1

        PS[J1, 1] = sin(2 * self.k * pi * abs(X1[J1] - 2) + pi)
        PS[J2, 1] = sin(2 * pi * abs(X1[J2] - 2) + pi)

        return PS


class MMF1e(MMF):
    """
    The MMF1_e test problem.

    Parameters
    ----------
    a: a > 0 and a <= 1. a controls the amplitude of the global Pareto set in x_1 in [2, 3]
    """
    def __init__(self, a=e):
        assert (a > 0 and a != 1), "Invalid parameter a!"
        self.a = a  # a controls the amplitude of the global PS in X1 in [1, 2)
        super().__init__(
            n_var=2, n_obj=2, n_constr=0, type_var=double,
            xl=[1, -1], xu=[3, 1]
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = zeros((len(X), self.n_var))
        X1 = X[:, 0]
        X2 = X[:, 1]

        # segments
        J1 = (1 <= X1) & (X1 < 2)
        J2 = (2 <= X1) & (X1 <= 3)

        F[:, 0] = abs(X1 - 2)
        F[J1, 1] = 1 - sqrt(abs(X1[J1] - 2)) + 2 * (X2[J1] - sin(6 * pi * abs(X1[J1] - 2) + pi)) ** 2
        F[J2, 1] = 1 - sqrt(abs(X1[J2] - 2)) + 2 * (
                X2[J2] - (self.a ** X1[J2]) * sin(6 * pi * abs(X1[J2] - 2) + pi)) ** 2
        out["F"] = F

    def _calc_pareto_set(self, n_pareto_points=500):
        PS = zeros((n_pareto_points, self.n_var))
        X1 = linspace(1, 3, n_pareto_points)

        # segments
        J1 = (1 <= X1) & (X1 < 2)
        J2 = (2 <= X1) & (X1 <= 3)

        PS[:, 0] = X1

        PS[J1, 1] = sin(6 * pi * abs(X1[J1] - 2) + pi)
        PS[J2, 1] = (self.a ** X1[J2]) * sin(6 * pi * abs(X1[J2] - 2) + pi)

        return PS
