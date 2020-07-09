import json
import numbers

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, cos
from numpy.linalg import inv, lstsq

from pymoo.problems.multi.distance_minimization import DistanceMinimization
from pymoo.util.misc import all_combinations
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class GeneralPolygonProblem(DistanceMinimization):
    def __init__(self, obj_points, basis, xl, xu):
        assert (len(basis) == 2), "The basis should only contain two vectors!"
        self.basis = basis

        self.obj_points_2d = obj_points
        self.obj_points = []

        # map polygon vertices into a 2-d subspace.
        for j, points in enumerate(obj_points):
            self.obj_points.append(self.map_2d_subspace(points))

        super().__init__(
            self.obj_points,
            xl=xl, xu=xu
        )

    def _calc_pareto_set(self, n_steps=100):
        V = []
        for vertices in self.obj_points_2d:
            V.extend(vertices)
        xl, yl = np.min(V, axis=0)
        xu, yu = np.max(V, axis=0)

        # Uniformly sample solutions on the plane, and remove the dominated solutions
        X_ = np.linspace(xl, xu, n_steps)
        Y_ = np.linspace(yl, yu, n_steps)

        X = all_combinations(X_, Y_)
        Xnd = self.map_2d_subspace(X)  # map the points to 2-dimensional subspace
        F = self.evaluate(Xnd, return_values_of=["F"])
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)

        return Xnd[nds]

    def _calc_pareto_front(self, n_steps=100):
        PS = self.pareto_set(n_steps)
        return self.evaluate(PS, return_values_of=["F"])

    def map_2d_subspace(self, X):
        return X @ self.basis

    def project_2d(self, X):
        B = self.basis.T  # basis
        # The projection matrix to project X to the plane defined by B
        P = B @ inv(B.T @ B) @ B.T
        Y = np.zeros((len(X), 2))
        for i, x in enumerate(X):
            # Project x to the plane
            Px = P @ x.T
            # Perform coordinate transformation, transform the solutions on the 2-dimensional subspace to the
            # 2-dimensional cartesian space (i.e., a plane defined by two unit vectors (1, 0), (0, 1). we can achieve
            # this by solving the linear equation: B * y = Px, where y is the transformed solutions
            Y[i] = lstsq(B, Px, rcond=-1)[0]
        return Y

    def visualize(self, X=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.tight_layout()
        ax.set_aspect('equal', 'box')
        ax.set(title=f"Decision Space ($D={self.n_var}$)", xlabel="$x_1$", ylabel="$x_2$")

        for i, V in enumerate(self.obj_points):
            # project v into 2-d plane
            v2d = self.project_2d(V)
            ax.scatter(v2d[:, 0], v2d[:, 1], s=40)

        # project X in to 2-d plane
        if X is not None:
            X2d = self.project_2d(X)
            ax.scatter(X2d[:, 0], X2d[:, 1], s=40, facecolors='none', edgecolors='black', label='Solutions')
            ax.legend()

        plt.show()


def create_regular_polygon_problem(centers, n_vertex, radii, xl, xu, basis=None):
    """
    Helper function to easily create regular polygon test problems.
    :param centers: numpy.ndarray
        location of polygon centers. It should be 2-dimensional.
    :param n_vertex: numpy.ndarray or int
        number of vertices of each polygon. All polygons have the same number of vertices if n_vertex in an int.
    :param radii: numpy.ndarray or number
        radius of each polygon. All polygons have the same radius if radii in an int.
    :param xl: numpy.ndarray or number
        lower bound
    :param xu: numpy.ndarray or number
        upper bound
    :param basis: numpy.ndarray
        two basis for plane transformation, default is [0, 1] and [1, 0].
    :return:
    """
    if basis is None:
        basis = np.array([[0, 1], [1, 0]])
    n_var = basis.shape[1]

    if isinstance(n_vertex, int):
        n_vertex = np.full(len(centers), n_vertex)

    if isinstance(radii, numbers.Number):
        radii = np.full(len(centers), radii)

    if isinstance(xl, numbers.Number):
        xl = np.full(n_var, xl)

    if isinstance(xu, numbers.Number):
        xu = np.full(n_var, xu)

    obj_points = [[] for _ in range(max(n_vertex))]
    for k, center in enumerate(centers):
        angles = (2 * pi * np.arange(n_vertex[k]) / n_vertex[k]).reshape(-1, 1)  # to column vector
        vertices = radii[k] * np.hstack((-sin(angles), cos(angles))) + center
        for j in range(n_vertex[k]):
            obj_points[j].append(vertices[j])
    return GeneralPolygonProblem(obj_points, basis, xl, xu)


def load_polygon_problem(file_path):
    with open(file_path) as f:
        data = json.load(f)
        # convert to nd array
        for k, v in data.items():
            data[k] = np.asarray(v)
        return GeneralPolygonProblem(**data)


def save_polygon_problem(problem, file_path):
    data = {
        "obj_points": problem.obj_points_2d.tolist(),
        "basis": problem.basis.tolist(),
        "xl": problem.xl.tolist(),
        "xu": problem.xu.tolist()
    }
    with open(file_path, 'w') as f:
        json.dump(data, f)
