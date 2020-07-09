from numpy import zeros, pi, linspace, sin, double, vstack

from pymoo.problems.multi.mmf import MMF


class MMF4(MMF):
    def __init__(self):
        super().__init__(
            n_var=2, n_obj=2, n_constr=0, type_var=double,
            xl=[-1, 0], xu=[1, 2]
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = zeros((len(X), self.n_var))
        X1 = X[:, 0]
        X2 = X[:, 1]

        # segments
        J1 = (0 <= X2) & (X2 < 1)
        J2 = (1 <= X2) & (X2 <= 2)

        F[:, 0] = abs(X1)
        F[J1, 1] = 1 - X1[J1] ** 2 + 2 * (X2[J1] - sin(pi * abs(X1[J1]))) ** 2
        F[J2, 1] = 1 - X1[J2] ** 2 + 2 * (X2[J2] - 1 - sin(pi * abs(X1[J2]))) ** 2

        out["F"] = F

    def _calc_pareto_set(self, n_pareto_points=500):
        h = int(n_pareto_points / 2)
        X1 = linspace(-1, 1, h)

        PS1 = zeros((h, self.n_var))
        PS1[:, 0] = X1
        PS1[:, 1] = sin(pi * abs(X1))

        PS2 = PS1.copy()
        PS2[:, 1] = PS1[:, 1] + 1

        return vstack((PS1, PS2))
