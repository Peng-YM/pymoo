import numpy as np
from numpy import exp, log10, sin, pi

from pymoo.problems.multi.mmf import MMF
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class MMF12(MMF):
    """
    The MMF12 test problem.

    Parameters
    ----------
    num_total_sets: the total number of global and local PSs.
    num_pieces: the number of discontinuous piece in each PF (PS).
    """
    def __init__(self, num_total_sets=2, num_pieces=4):
        self.np = num_total_sets
        self.q = num_pieces
        super().__init__(
            n_var=2, n_obj=2, n_constr=0, type_var=np.double,
            xl=[0, 0], xu=[1, 1]
        )

    def _evaluate(self, X, out, *args, **kwargs):
        X1 = X[:, 0]
        X2 = X[:, 1]
        F = X.copy()
        g = 2 - exp(-2 * log10(2) * ((X2 - 0.1) / 0.8) ** 2) * (sin(self.np * pi * X2)**6)
        h = 1 - (X1 / g) ** 2 - (X1 / g) * sin(2 * pi * self.q * X1)
        F[:, 1] = g * h

        out["F"] = F

    def _calc_pareto_set(self, n_pareto_points=1000):
        X = np.zeros((n_pareto_points, 2))
        X[:, 0] = np.linspace(0, 1, n_pareto_points)
        X[:, 1] = 1 / (2 * self.np)

        # Calculate the values in the objective space, remove dominated solutions
        F = self.evaluate(X, return_values_of=["F"])
        # Non-dominated sorting
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        return X[nds, :]
