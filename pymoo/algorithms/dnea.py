import numpy as np
from numpy import max, zeros, ones
from scipy.spatial.distance import cdist

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class DNEA(GeneticAlgorithm):
    def __init__(self,
                 sigma_obj,
                 sigma_var,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=RandomSelection(),
                 crossover=SimulatedBinaryCrossover(eta=20, prob=1),
                 mutation=PolynomialMutation(prob=0.5, eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=DoubleNichedSurvival(sigma_obj, sigma_var),
            eliminate_duplicates=eliminate_duplicates,
            n_offsprings=n_offsprings,
            display=display,
            **kwargs
        )


class DoubleNichedSurvival(Survival):
    def __init__(self, sigma_obj, sigma_var):
        self.sorting = NonDominatedSorting(method="efficient_non_dominated_sort")
        self.sigma_obj = sigma_obj
        self.sigma_var = sigma_var
        super().__init__()

    def _do(self, problem, pop, n_survive, **kwargs):
        F = pop.get("F")
        fronts = self.sorting.do(F, n_stop_if_ranked=n_survive)

        # the final indices of surviving individuals
        survivors = []

        for front in fronts:
            if len(front) + len(survivors) > n_survive:
                # this is the last front to be included
                n_remains = n_survive - len(survivors)
                while len(front) > n_remains:
                    pop_front = pop[front]
                    # calculate the double sharing function
                    fitness = [self.double_sharing(pop_front, indv) for indv in pop_front]
                    # remove the worst one
                    worst_i = np.argmax(fitness)
                    front = np.delete(front, worst_i)

            survivors.extend(front)
        return pop[survivors]

    def double_sharing(self, pop_front, indv):
        sz = len(pop_front)

        # objective and variable distance from indv to other individuals in the current front
        dF = cdist(np.asarray([indv.F]), pop_front.get("F"))[0]
        dX = cdist(np.asarray([indv.X]), pop_front.get("X"))[0]

        Sh_obj = max(np.vstack((zeros(sz), ones(sz) - dF / self.sigma_obj)), axis=0)
        Sh_var = max(np.vstack((zeros(sz), ones(sz) - dX / self.sigma_var)), axis=0)
        return sum(Sh_obj + Sh_var)
