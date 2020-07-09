from numpy import sqrt, sin, zeros, pi, linspace, cos, double, vstack

from pymoo.problems.multi.mmf import MMF


class MMF7(MMF):
    def __init__(self):
        super().__init__(
            n_var=2, n_obj=2, n_constr=0, type_var=double,
            xl=[1, -1], xu=[3, 1]
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = zeros((len(X), self.n_var))
        X1 = X[:, 0]
        X2 = X[:, 1]

        F[:, 0] = abs(X1 - 2)
        F[:, 1] = 1 - sqrt(abs(X1 - 2)) + (
                X2 - (0.3 * abs(X1 - 2) ** 2 * cos(24 * pi * abs(X1 - 2) + 4 * pi) + 0.6 * abs(X1 - 2)) * sin(
            6 * pi * abs(X1 - 2) + pi)) ** 2;

        out["F"] = F

    def _calc_pareto_set(self, n_pareto_points=500):
        X1 = linspace(1, 3, n_pareto_points)
        X2 = (0.3 * abs(X1 - 2) ** 2 * cos(24 * pi * abs(X1 - 2) + 4 * pi) + 0.6 * abs(X1 - 2)) * sin(
            6 * pi * abs(X1 - 2) + pi)
        return vstack((X1, X2)).T
