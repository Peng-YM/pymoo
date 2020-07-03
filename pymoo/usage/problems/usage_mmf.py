# This import registers the 3D projection, but is otherwise unused.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize
from pymoo.problems.multi.mmf import *
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.visualization.pcp import PCP
from pymoo.visualization.scatter import Scatter


# visualize a problem
def visualize_solutions(X, F, PS=None, PF=None, title=None, save_to=None):
    n_var, n_obj = X.shape[1], F.shape[1]
    with plt.style.context(['seaborn-paper']):
        plt.rc('font', family='serif')
        fig = plt.figure(figsize=(9 * 1.1, 4 * 1.1))
        if title:
            fig.suptitle(title)
        # Decision Space
        if n_var == 2:
            ax1 = fig.add_subplot(121)
            ax1.set(xlabel="$x_1$", ylabel="$x_2$")

            plot = Scatter(ax=ax1)
            if PS is not None:
                plot.add(PS, s=10, facecolors='none', edgecolors='r', label="Pareto Set")
            if X is not None:
                plot.add(X, s=30, color='b', label="Solutions")
        elif n_var == 3:
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$")

            plot = Scatter(ax=ax1)
            if PS is not None:
                plot.add(PS, s=10, facecolor=(1, 1, 1, 1), edgecolors='r', label="Pareto Set")
            if X is not None:
                plot.add(X, s=30, color='b', label="Solutions")
        else:
            ax1 = fig.add_subplot(121)
            plot = PCP(ax=ax1)
            plot.set_axis_style(color="grey", alpha=1)
            if PS is not None:
                plot.add(PS, color="grey")
            if X is not None:
                plot.add(X, color="blue")

        plot.do()
        ax1.set_title("Decision Space")
        ax1.legend()

        # Objective Space
        if n_obj == 2:
            ax2 = fig.add_subplot(122)
            ax2.set(xlabel="$f_1$", ylabel="$f_2$")
            plot = Scatter(ax=ax2)

            if PF is not None:
                plot.add(PF, s=10, facecolors='none', edgecolors='r', label="Pareto Front")
            if F is not None:
                plot.add(F, s=30, color='b', label="Solutions")
        elif n_obj == 3:
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set(xlabel="$f_1$", ylabel="$f_2$", zlabel="$f_3$")
            plot = Scatter(ax=ax2)

            if PF is not None:
                plot.add(PF, s=10, facecolor=(0, 0, 0, 0), edgecolors='r', label="Pareto Front")
            if F is not None:
                plot.add(F, s=30, color='b', label="Solutions")
        else:
            ax2 = fig.add_subplot(122)
            plot = PCP(ax=ax2)
            plot.set_axis_style(color="grey", alpha=1)
            if PF is not None:
                plot.add(PF, color="grey")
            if F is not None:
                plot.add(F, color="blue")

        plot.do()
        ax2.set_title("Objective Space")
        ax2.legend()

        if save_to is not None:
            plt.savefig(f"{save_to}.pdf")
        else:
            plt.show()


if __name__ == '__main__':
    # create the reference directions to be used for the optimization
    problem = MMF13(num_total_sets=2)
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=40)
    PS = problem.pareto_set(500)
    PF = problem.pareto_front(500)

    nds = NonDominatedSorting().do(PF, only_non_dominated_front=True)
    assert len(nds) == len(PF), "Incorrect PF!"

    algorithm = NSGA2(pop_size=100)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 1000),
                   seed=1,
                   verbose=False)

    fig_name = f"{algorithm.__class__.__name__} on {problem.__class__.__name__}"
    visualize_solutions(res.X, res.F, PS, PF, title=fig_name)
