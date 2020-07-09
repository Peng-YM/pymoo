import numpy as np
from numpy import cos, sin, pi, sqrt, exp, log10

from pymoo.model.problem import Problem


class MMF14Base(Problem):
    def __init__(self, g, n_var, n_obj, num_pareto_sets=2):
        self.np = num_pareto_sets
        self.g = g
        super().__init__(
            n_var=n_var, n_obj=n_obj, n_constr=0, type_var=np.double,
            xl=np.full(n_var, 0), xu=np.full(n_var, 1),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        M = self.n_obj
        N = len(X)  # number of individuals
        g = np.tile(self.g(X), (M, 1)).T  # g --> N x M matrix
        Cos = np.hstack((np.ones((N, 1)), cos(X[:, :M - 1] * pi / 2)))
        Sin = np.hstack((np.ones((N, 1)), sin(X[:, M - 2::-1] * pi / 2)))
        out["F"] = g * np.fliplr(np.cumprod(Cos, axis=1)) * Sin

    def _calc_pareto_front(self, n_pareto_points=500):
        PS = self._calc_pareto_set(n_pareto_points)
        return self.evaluate(PS, return_values_of=["F"])


class MMF14(MMF14Base):
    """
    The MMF14 test problem.

    Parameters
    ----------
    num_pareto_sets: the total number of global PSs.
    """

    def __init__(self, num_pareto_sets=2):
        self.np = num_pareto_sets
        super().__init__(
            g=lambda X: 2 - sin(self.np * pi * X[:, -1]) ** 2,
            n_var=3, n_obj=3, num_pareto_sets=num_pareto_sets)

    def _calc_pareto_set(self, n_pareto_points=500):
        PS = []

        h = int(sqrt(n_pareto_points / self.np))
        X1 = np.linspace(0, 1, h)
        X2 = np.linspace(0, 1, h)

        # Generate a mesh grid from X1 and X2
        grid = np.meshgrid(X1, X2)
        XX = np.array(grid).T.reshape(-1, 2)

        for i in range(self.np):
            # The i-th Pareto subset
            PSi = np.zeros((len(XX), 3))
            PSi[:, 0] = XX[:, 0]
            PSi[:, 1] = XX[:, 1]
            PSi[:, 2] = (1 / (2 * self.np)) + i / self.np
            PS.append(PSi)

        return np.vstack(PS)


class MMF14a(MMF14Base):
    """
    The MMF14_a test problem.

    Parameters
    ----------
    num_pareto_sets: the total number of global PSs.
    """

    def __init__(self, num_pareto_sets=2):
        self.np = num_pareto_sets
        super().__init__(
            g=self.g,
            n_var=3, n_obj=3, num_pareto_sets=num_pareto_sets)

    def g(self, X):
        Xg = X[:, 2] - 0.5 * sin(pi * X[:, 1])
        return 2 - sin(self.np * pi * (Xg + 1 / (2 * self.np))) ** 2

    def _calc_pareto_set(self, n_pareto_points=500):
        PS = []

        h = int(sqrt(n_pareto_points / self.np))
        X1 = np.linspace(0, 1, h)
        X2 = np.linspace(0, 1, h)

        # Generate a mesh grid from X1 and X2
        grid = np.meshgrid(X1, X2)
        XX = np.array(grid).T.reshape(-1, 2)

        for i in range(self.np):
            # The i-th Pareto subset
            PSi = np.zeros((len(XX), 3))
            PSi[:, 0] = XX[:, 0]
            PSi[:, 1] = XX[:, 1]
            PSi[:, 2] = 0.5 * sin(pi * PSi[:, 1]) + i / self.np
            PS.append(PSi)

        return np.vstack(PS)


class MMF15(MMF14Base):
    """
    The MMF15 test problem.

    Parameters
    ----------
    num_total_sets: the total number of local and global PSs.
    """

    def __init__(self, num_total_sets=2):
        self.np = num_total_sets
        super().__init__(
            g=lambda X: 2 - exp(-2 * log10(2) * ((X[:, -1] - 0.1) / 0.8) ** 2) * sin(self.np * pi * X[:, -1]) ** 2,
            n_var=3, n_obj=3, num_pareto_sets=num_total_sets)

    def _calc_pareto_set(self, n_pareto_points=500):
        h = int(sqrt(n_pareto_points))
        X1 = np.linspace(0, 1, h)
        X2 = np.linspace(0, 1, h)

        # Generate a mesh grid from X1 and X2
        grid = np.meshgrid(X1, X2)
        XX = np.array(grid).T.reshape(-1, 2)
        PS = np.zeros((len(XX), 3))

        PS[:, 0] = XX[:, 0]
        PS[:, 1] = XX[:, 1]
        PS[:, 2] = 1 / (2 * self.np)

        return PS


class MMF15a(MMF14Base):
    """
    The MMF15_a test problem.

    Parameters
    ----------
    num_total_sets: the total number of local and global PSs.
    """

    def __init__(self, num_total_sets=2):
        self.np = num_total_sets
        super().__init__(
            g=self.g,
            n_var=3, n_obj=3, num_pareto_sets=num_total_sets)

    def g(self, X):
        t = -0.5 * sin(pi * X[:, -2]) + X[:, -1]
        return 2 - exp(-2 * log10(2) * ((t + 1 / (2 * self.np) - 0.1) / 0.8) ** 2) * sin(
            self.np * pi * (t + 1 / (2 * self.np))) ** 2

    def _calc_pareto_set(self, n_pareto_points=500):
        h = int(sqrt(n_pareto_points))
        X1 = np.linspace(0, 1, h)
        X2 = np.linspace(0, 1, h)

        # Generate a mesh grid from X1 and X2
        grid = np.meshgrid(X1, X2)
        XX = np.array(grid).T.reshape(-1, 2)
        PS = np.zeros((len(XX), 3))

        PS[:, 0] = XX[:, 0]
        PS[:, 1] = XX[:, 1]
        PS[:, 2] = 0.5 * sin(pi * PS[:, 1])

        return PS
