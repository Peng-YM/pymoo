from pymoo.algorithms.moead_ad import MOEAD_AD
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize
from pymoo.problems.multi.mmf import MMF2
from pymoo.visualization.scatter import Scatter

problem = MMF2()

algorithm = MOEAD_AD(
    ref_dirs=get_reference_directions('energy', n_dim=problem.n_obj, n_points=100)
)
res = minimize(problem, algorithm, termination=('n_eval', 10000), verbose=True)
Scatter().add(res.X).show()
