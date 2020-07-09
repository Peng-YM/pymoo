import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.problem import Problem


class DistanceMinimization(Problem):
    def __init__(self, obj_points, xl, xu, dist_metric='euclidean'):
        self.obj_points = obj_points
        self.metric = dist_metric
        super().__init__(
            n_obj=len(obj_points), n_var=len(xl),
            xl=xl, xu=xu
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.full((len(X), self.n_obj), float('inf'))
        # evaluate each objective
        for j, points in enumerate(self.obj_points):
            F[:, j] = np.min(cdist(X, points, self.metric), axis=1)
        out['F'] = F
