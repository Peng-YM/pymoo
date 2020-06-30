import matplotlib.pyplot as plt

from pymoo.model.population import Population
from pymoo.problems.multi.sympart import SYMPARTRotated
from pymoo.subset_selection.two_stage_subset_selection import TwoStageSubsetSelection, DBCANClustering, \
    ModifiedTwoStageSubsetSelection
from pymoo.visualization.scatter import Scatter

n_candidates = 3000
problem = SYMPARTRotated()
PS = problem.pareto_set(n_candidates)
PF = problem.evaluate(PS, return_values_of=["F"])

pop = Population(len(PS))
pop.new()
pop.set("X", PS, "F", PF)

for method, name in [(TwoStageSubsetSelection, "TSS"), (ModifiedTwoStageSubsetSelection, "$TSS^+$")]:
    clustering = DBCANClustering(epsilon=0.05, min_samples=5)
    selection = method(clustering, delta_max=0.03)

    selected = selection.do(pop, n_select=50)

    # visualization
    X = PS[selected]
    F = PF[selected]

    with plt.style.context(['seaborn-paper']):
        plot = Scatter()
        plot.add(PS, color='b', s=5, label="Pareto set")
        plot.add(X, facecolor='none', edgecolor='r', s=40, label="Selected solutions", linewidth=2)
        plot.do()
        plt.legend()

        plot = Scatter()
        plot.add(PF, color='b', s=5, label="Pareto set")
        plot.add(F, facecolor='none', edgecolor='r', s=40, label="Selected solutions", linewidth=2)
        plot.do()
        plt.legend()
        plt.title(name)
        plt.show()
