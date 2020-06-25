import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from pymoo.algorithms.moead_mm import MOEAD_MM
from pymoo.factory import get_reference_directions
from pymoo.model.evaluator import Evaluator
from pymoo.optimize import minimize
from pymoo.problems.multi.distance_minimization import DistanceMinimization
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.visualization.scatter import Scatter

polyhedra = np.array([
    (0, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (1, 0, 0),
    (0.5, 0.5, 0.707)
])

shifts = np.array([
    [0, 0, 0], [0, 5, 0], [5, 0, 0], [5, 5, 0],
    [0, 0, 5], [0, 5, 5], [5, 0, 5], [5, 5, 5]
])

obj_points = [[] for _ in range(5)]  # 5-objective
for s in shifts:
    vertices = polyhedra + s
    for j, v in enumerate(vertices):
        obj_points[j].append(v)

problem = DistanceMinimization(
    obj_points=obj_points,
    xl=np.full(3, -100),
    xu=np.full(3, 100)
)

uea_evaluator = Evaluator(use_archive=True)

res = minimize(
    problem=problem,
    algorithm=MOEAD_MM(
        ref_dirs=get_reference_directions('energy', problem.n_obj, 100),
        sub_pop_size=4
    ),
    termination=('n_gen', 200),
    verbose=True,
    evaluator=uea_evaluator
)

# non-dominated-sorting
archive = res.algorithm.evaluator.archive
selected = NonDominatedSorting().do(archive.get("F"), only_non_dominated_front=True)

# visualization
fig = plt.figure()
plt.rc('font', family='serif')
ax = fig.add_subplot(111, projection='3d')
ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title="Decision Space")
plot = Scatter(ax=ax)

# show polyhedras
for s in shifts:
    vertices = polyhedra + s
    x = [vertices[0, 0], vertices[1, 0], vertices[2, 0], vertices[3, 0], vertices[0, 0]]
    y = [vertices[0, 1], vertices[1, 1], vertices[2, 1], vertices[3, 1], vertices[0, 1]]
    z = [vertices[0, 2], vertices[1, 2], vertices[2, 2], vertices[3, 2], vertices[0, 2]]
    ax.plot(x, y, z, color='black')

    for k in range(4):
        x = [vertices[k, 0], vertices[4, 0]]
        y = [vertices[k, 1], vertices[4, 1]]
        z = [vertices[k, 2], vertices[4, 2]]
        ax.plot(x, y, z, color='black')

# show solutions
plot.add(archive[selected].get("X"), s=30, facecolor=(1, 1, 1, 1), edgecolors='r', label="Solutions").do()

ax.legend()
plt.show()
