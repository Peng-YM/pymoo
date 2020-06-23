import numpy as np
from scipy.spatial.distance import pdist, squareform
from pymoo.model.population import Population
from pymoo.algorithms.moead import MOEAD


class MOEAD_MM(MOEAD):
    def __init__(self, ref_dirs, sub_pop_size, **kwargs):
        self.ref_dirs = ref_dirs
        self.mu = sub_pop_size
        super().__init__(
            ref_dirs=ref_dirs,
            pop_size=sub_pop_size * len(ref_dirs),
            n_neighbors=int(len(ref_dirs) / 10),
            **kwargs
        )

    def _initialize(self):
        super()._initialize()
        # randomly assign mu solutions to each sub-population
        self.sub_pops = []
        for i, _ in enumerate(self.ref_dirs):
            self.sub_pops.append(self.pop[i * self.mu: (i+1) * self.mu])

    def _next(self):
        # estimate the clearing radius
        clearing_radius = self.estimate_clearing_radius()
        # update each sub-population
        for i, sub_pop in enumerate(self.sub_pops):
            offspring = self.mating_(i)
            self.sub_pops[i] = self.selection_(sub_pop.merge(offspring), self.ref_dirs[i], clearing_radius)
        # merge sub-populations
        self.pop = Population.create(*self.sub_pops)

    def mating_(self, sub_pop_index):
        repair, crossover, mutation = self.repair, self.mating.crossover, self.mating.mutation
        x1 = np.random.choice(self.sub_pops[sub_pop_index])  # current sub-population
        # randomly select a neighborhood weight vector j
        j = np.random.choice(self.neighbors[sub_pop_index])
        x2 = np.random.choice(self.sub_pops[j])
        # produce offspring from x1 and x2
        off = crossover.do(self.problem, Population.create(x1, x2), np.array([[0, 1]]))
        off = mutation.do(self.problem, off)
        # repair the offspring if necessary
        off = repair.do(self.problem, Population.create(off), algorithm=self)[0]
        # evaluate the offspring
        self.evaluator.eval(self.problem, off)
        # update the ideal point
        self.ideal_point = np.min(np.vstack([self.ideal_point, off.F]), axis=0)
        return off

    def selection_(self, C, ref_dir, clearing_radius):
        X = np.asarray([indv.X for indv in C])
        F = np.asarray([indv.X for indv in C])
        D = squareform(pdist(X))
        # find the closest pair of points in C
        min_d, min_i, min_j = float("inf"), 0, 0
        for i in range(len(C)-1):
            for j in range(i+1, len(C)):
                if D[i, j] < min_d:
                    min_d = D[i, j]
                    min_i, min_j = i, j
        # calculate the scalarizing function values
        FV = self._decomposition.do(F, weights=ref_dir, ideal_point=self.ideal_point)
        # clearing
        if min_d < clearing_radius:
            # compare Ci and Cj with scalarizing function values
            i = min_i if FV[min_i] > FV[min_j] else min_j
        else:
            i = np.argmax(FV)
        # remove one solutions
        return np.delete(C, i)

    def estimate_clearing_radius(self, percentage=0.1):
        k = int(self.pop_size * percentage)
        # calculate pairwise distance in the decision space
        D = squareform(pdist(self.pop.get("X")))
        # sort D to find the distance from each solution to its k-th nearest neighbor
        k_dist = np.sort(D, axis=1)[:, k]
        # the clearing radius is estimated as the average of k_dist
        return np.mean(k_dist)
