from pymoo.algorithms.dnea import DNEA
from pymoo.problems.multi.mmf import MMF2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


problem = MMF2()
algorithm = DNEA(sigma_obj=0.05, sigma_var=0.3)

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 100),
    verbose=True
)
Scatter().add(res.X).show()

