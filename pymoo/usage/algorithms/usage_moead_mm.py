from pymoo.algorithms.moead_mm import MOEAD_MM
from pymoo.problems.multi.mmf.mmf1 import MMF1
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize


problem = MMF1()
algorithm = MOEAD_MM(
    ref_dirs=get_reference_directions('energy', n_dim=problem.n_obj, n_points=100),
    sub_pop_size=4
)
res = minimize(problem, algorithm, termination=('n_gen', 100), verbose=True)
