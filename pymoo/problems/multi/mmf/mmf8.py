from numpy import zeros, pi, linspace, sin, double, vstack, sqrt

from pymoo.problems.multi.mmf import MMF


class MMF8(MMF):
    def __init__(self):
        super().__init__(
            n_var=2, n_obj=2, n_constr=0, type_var=double,
            xl=[-pi, 0], xu=[pi, 9]
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = zeros((len(X), self.n_var))
        X1 = X[:, 0]
        X2 = X[:, 1]
        J1 = (0 <= X2) & (X2 <= 4)
        J2 = (4 <= X2) & (X2 <= 9)

        F[:, 0] = sin(abs(X1))
        F[J1, 1] = sqrt(1 - (sin(abs(X1[J1]))) ** 2) + 2 * (X2[J1] - sin(abs(X1[J1])) - abs(X1[J1])) ** 2
        F[J2, 1] = sqrt(1 - (sin(abs(X1[J2]))) ** 2) + 2 * (X2[J2] - 4 - sin(abs(X1[J2])) - abs(X1[J2])) ** 2
        out["F"] = F

    def _calc_pareto_set(self, n_pareto_points=500):
        h = int(n_pareto_points / 2)
        X1 = linspace(-pi, pi, h)

        PS1 = zeros((h, self.n_var))
        PS1[:, 0] = X1
        PS1[:, 1] = sin(abs(X1)) + abs(X1)

        PS2 = PS1.copy()
        PS2[:, 1] = PS1[:, 1] + 4

        return vstack((PS1, PS2))
