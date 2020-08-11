from pymoo.algorithms.omni_optimizer import OmniOptimizer
from pymoo.optimize import minimize
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.visualization.scatter import Scatter

POP_SIZE = 200

problem = OmniTest()
algorithm = OmniOptimizer(
    pop_size=POP_SIZE
)

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 300),
    verbose=True
)
Scatter().add(res.X).show()
