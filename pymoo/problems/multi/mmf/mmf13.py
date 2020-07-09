import numpy as np
from numpy import exp, log10, sin, pi, sqrt

from pymoo.model.problem import Problem


class MMF13(Problem):
    """
    The MMF13 test problem.

    Parameters
    ----------
    num_total_sets: the total number of global and local PSs.
    """

    def __init__(self, num_total_sets=2):
        self.np = num_total_sets
        super().__init__(
            n_var=3, n_obj=2, n_constr=0, type_var=np.double,
            xl=[0.1, 0.1, 0.1], xu=[1.1, 1.1, 1.1]
        )

    def _evaluate(self, X, out, *args, **kwargs):
        X1, X2, X3 = X[:, 0], X[:, 1], X[:, 2]
        F = np.zeros((len(X), 2))
        F[:, 0] = X1
        t = X2 + sqrt(X3)
        g = 2 - exp(-2 * log10(2) * ((t - 0.1) / 0.8) ** 2) * sin(self.np * pi * t) ** 6
        F[:, 1] = g / X1

        out["F"] = F

    def _calc_pareto_front(self, n_pareto_points=500):
        PF = np.zeros((n_pareto_points, 2))
        F1 = np.linspace(0.1, 1.1, n_pareto_points)
        PF[:, 0] = F1
        PF[:, 1] = (2 - exp(-2 * log10(2) * ((1 / (2 * self.np) - 0.1) / 0.8) ** 2) * (sin(
            pi / 2) ** 6)) / F1

        # PF = self.evaluate(self.pareto_set(n_pareto_points), return_values_of=['F'])
        return PF

    def _calc_pareto_set(self, n_pareto_points=500):
        h = int(sqrt(n_pareto_points))
        PS = np.zeros((h ** 2, 3))

        X1 = np.linspace(0.1, 1.1, h)
        X2 = np.linspace(0.1, 1 / (2 * self.np) - sqrt(0.1), h)

        # Generate a mesh grid from X1 and X2
        grid = np.meshgrid(X1, X2)
        XX = np.array(grid).T.reshape(-1, 2)

        PS[:, 0] = XX[:, 0]
        PS[:, 1] = XX[:, 1]
        PS[:, 2] = (1 / (2 * self.np) - XX[:, 1]) ** 2

        return PS
