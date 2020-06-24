import numpy as np

from pymoo.algorithms.moead_mm import MOEAD_MM
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize
from pymoo.problems.multi.distance_minimization import create_regular_polygon_problem

centers = np.array([
    [0, 0], [0, 5], [5, 0], [5, 5]
])
problem = create_regular_polygon_problem(
    centers=centers,
    radii=[1, 0.5, 0.5, 1],
    n_vertex=[6, 6, 6, 6],
    xl=-100, xu=100
)

problem.visualize()

res = minimize(
    problem=problem,
    algorithm=MOEAD_MM(
        ref_dirs=get_reference_directions('energy', problem.n_obj, 100),
        sub_pop_size=4
    ),
    termination=('n_gen', 200),
    verbose=True
)
problem.visualize(res.X)
