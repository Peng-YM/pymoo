from numpy import sqrt, zeros, pi, linspace, sin, double, vstack, array

from pymoo.problems.multi.mmf import MMF


def f(x):
    x1, x2 = x[0], x[1]
    if (-1 < x2 <= 0) and ((7 / 6 < x1 <= 8 / 6) or (9 / 6 < x1 <= 10 / 6) or (11 / 6 < x1 <= 2)):
        x2 = x2
    elif (-1 < x2 <= 0) and ((2 < x1 <= 13 / 6) or (14 / 6 < x1 <= 15 / 6) or (16 / 6 < x1 <= 17 / 6)):
        x2 = x2
    elif abs(x2 - 1) < 1e-8 and abs(x1 - 1) < 1e-8:
        x2 = x2 - 1
    elif (1 < x2 <= 2) and ((1 < x1 <= 7 / 6) or (4 / 3 < x1 <= 3 / 2) or (5 / 3 < x1 <= 11 / 6)):
        x2 = x2 - 1
    elif (1 < x2 <= 2) and ((13 / 6 < x1 <= 14 / 6) or (15 / 6 < x1 <= 16 / 6) or (17 / 6 < x1 <= 3)):
        x2 = x2 - 1
    elif (0 < x2 <= 1) and ((1 < x1 <= 7 / 6) or (4 / 3 < x1 <= 3 / 2) or (5 / 3 < x1 <= 11 / 6) or (
            13 / 6 < x1 <= 14 / 6) or (15 / 6 < x1 <= 16 / 6) or (17 / 6 < x1 <= 3)):
        x2 = x2
    elif (0 < x2 <= 1) and ((7 / 6 < x1 <= 8 / 6) or (9 / 6 < x1 <= 10 / 6) or (11 / 6 < x1 <= 2) or (
            2 < x1 <= 13 / 6) or (14 / 6 < x1 <= 15 / 6) or (16 / 6 < x1 <= 17 / 6)):
        x2 = x2 - 1

    y1 = abs(x1 - 2)
    y2 = 1.0 - sqrt(abs(x1 - 2)) + 2.0 * (x2 - sin(6 * pi * abs(x1 - 2) + pi)) ** 2

    return array([y1, y2])


class MMF6(MMF):
    def __init__(self):
        super().__init__(
            n_var=2, n_obj=2, n_constr=0, type_var=double,
            xl=[1, -1], xu=[3, 2], elementwise_evaluation=True
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = f(x)

    def _calc_pareto_set(self, n_pareto_points=500):
        h = int(n_pareto_points / 2)
        X1 = linspace(1, 3, h)

        PS1 = zeros((h, self.n_var))
        PS1[:, 0] = X1
        PS1[:, 1] = sin(6 * pi * abs(X1 - 2) + pi)

        PS2 = PS1.copy()
        PS2[:, 1] = PS1[:, 1] + 1

        return vstack((PS1, PS2))
