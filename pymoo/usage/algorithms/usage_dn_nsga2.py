from pymoo.algorithms.dn_nsga2 import DN_NSGA2
from pymoo.optimize import minimize
from pymoo.problems.multi.mmf import *
from pymoo.visualization.scatter import Scatter

POP_SIZE = 200

problem = MMF2()
algorithm = DN_NSGA2(
    crowding_factor=5,
    pop_size=POP_SIZE
)

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 200),
    verbose=True
)
Scatter().add(res.X).show()
