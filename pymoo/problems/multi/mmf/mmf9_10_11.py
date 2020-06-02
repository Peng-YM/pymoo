from numpy import zeros, pi, linspace, double, arange, sin, exp, log10

from pymoo.problems.multi.mmf import MMF


class MMF9Base(MMF):
    def __init__(self, g):
        self.g = g

        super().__init__(
            n_var=2, n_obj=2, n_constr=0, type_var=double,
            xl=[0.1, 0.1], xu=[1.1, 1.1]
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = zeros((len(X), self.n_var))
        X1 = X[:, 0]
        X2 = X[:, 1]

        F[:, 0] = X1
        F[:, 1] = self.g(X2) / X1

        out["F"] = F


class MMF9(MMF9Base):
    """
    The MMF9 test problem.

    Parameters
    ----------
    num_pareto_sets: the number of global PSs, default is 2.
    """
    def __init__(self, num_pareto_sets=2):
        self.np = num_pareto_sets
        super().__init__(
            g=lambda X: 2 - sin(self.np * pi * X) ** 6,
        )

    def _calc_pareto_set(self, n_pareto_points=500):
        h = int(n_pareto_points / self.np)
        PS = zeros((h * self.np, 2))
        for i in range(self.np):
            I = arange(i * h, i * h + h)
            PS[I, 0] = linspace(0.1, 1.1, h)
            PS[I, 1] = (1 / (2 * self.np)) + (1 / self.np) * i
        return PS


class MMF10(MMF9Base):
    def __init__(self):
        super().__init__(
            g=lambda X: 2 - exp(-((X - 0.2) / 0.004) ** 2) - 0.8 * exp(-((X - 0.6) / 0.4) ** 2),
        )

    def _calc_pareto_set(self, n_pareto_points=500):
        PS = zeros((n_pareto_points, 2))
        PS[:, 0] = linspace(0.1, 1.1, n_pareto_points)
        PS[:, 1] = 0.2

        return PS


class MMF11(MMF9Base):
    """
    The MMF11 test problem.

    Parameters
    ----------
    num_total_sets: the total number of global and local PSs.
    """
    def __init__(self, num_total_sets=2):
        self.np = num_total_sets
        super().__init__(
            g=lambda X: 2 - exp(-2 * log10(2) * ((X - 0.1) / 0.8) ** 2) * sin(self.np * pi * X) ** 6
        )

    def _calc_pareto_set(self, n_pareto_points=500):
        PS = zeros((n_pareto_points, 2))
        PS[:, 0] = linspace(0.1, 1.1, n_pareto_points)
        PS[:, 1] = 1 / (2 * self.np)

        return PS
