import matplotlib.pyplot as plt

from pymoo.model.population import Population
from pymoo.problems.multi.sympart import SYMPARTRotated
from pymoo.subset_selection.weight_vector_based_subset_selection import WeightVectorBasedSubsetSelection
from pymoo.visualization.scatter import Scatter

n_candidates = 3000
problem = SYMPARTRotated()
PS = problem.pareto_set(n_candidates)
PF = problem.evaluate(PS, return_values_of=["F"])

pop = Population(len(PS))
pop.new()
pop.set("X", PS, "F", PF)

selection = WeightVectorBasedSubsetSelection(pop)

selected = selection.do(n_select=50)

# visualization
X = PS[selected]
F = PF[selected]

plot = Scatter()
plot.add(PS, color='r', s=10, label="Pareto set")
plot.add(X, color='b', s=30, label="Selected solutions")
plot.do()
plt.legend()

plot = Scatter()
plot.add(PF, color='r', s=10, label="Pareto set")
plot.add(F, color='b', s=30, label="Selected solutions")
plot.do()
plt.legend()

plt.show()
