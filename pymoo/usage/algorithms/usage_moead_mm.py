from pymoo.algorithms.moead_mm import MOEAD_MM
from pymoo.problems.multi.mmf import MMF2
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize


problem = MMF2()
algorithm = MOEAD_MM(
    ref_dirs=get_reference_directions('energy', n_dim=problem.n_obj, n_points=100),
    sub_pop_size=4
)
res = minimize(problem, algorithm, termination=('n_gen', 100), verbose=True)
Scatter().add(res.X).show()