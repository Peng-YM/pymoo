import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from pymoo.algorithms.moead import MOEAD
from pymoo.model.population import Population
from pymoo.subset_selection.distance_based_subset_selection import DistanceBasedSubsetSelection
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class MOEAD_AD(MOEAD):
    def __init__(self, ref_dirs, neighborhood_size=0.1, **kwargs):
        super().__init__(
            ref_dirs,
            **kwargs
        )
        self.neighborhood_size = neighborhood_size

    def _initialize(self):
        super()._initialize()
        self.sub_pops = [[indv] for indv in self.pop]

    def _next(self):
        # iteratively produce N offsprings
        for _ in range(self.pop_size):
            # bounds for normalization
            self.F_min = np.min(self.pop.get("F"), axis=0)
            self.F_max = np.max(self.pop.get("F"), axis=0)
            self.X_min = np.min(self.pop.get("X"), axis=0)
            self.X_max = np.max(self.pop.get("X"), axis=0)

            # randomly select two individuals
            mu = len(self.pop)
            r1, r2 = self.pop[np.random.choice(mu, 2, replace=False)]

            # produce an offspring
            repair, crossover, mutation = self.repair, self.mating.crossover, self.mating.mutation
            off = crossover.do(self.problem, r1, r2)
            off = mutation.do(self.problem, off)
            off = repair.do(self.problem, off, algorithm=self)
            off = off[0]

            # evaluate the offspring
            self.evaluator.eval(self.problem, off)

            # update the ideal point
            self.ideal_point = np.min(np.vstack([self.ideal_point, off.F]), axis=0)

            # update the sub-populations
            self.update(off)

            # merge the sub-populations
            self.pop = Population.create(*[indv for sub in self.sub_pops for indv in sub])

        # select sparse solutions in the decision space before termination
        if self.termination.has_terminated(algorithm=self):
            self.pop = self.select_sparse_solutions()

    def update(self, off):
        # normalize the objectives and decision variables of the offspring
        f = (off.F - self.F_min) / (self.F_max - self.F_min)
        x = (off.X - self.X_min) / (self.X_max - self.X_min)

        # assign the offspring to a weight vector j with smallest perpendicular distance
        PD = cdist(np.array([f]), self.ref_dirs, metric='cosine')[0]
        j = int(np.argmin(PD))
        fitness = lambda a: self._decomposition.do(a.F, weights=self.ref_dirs[j], ideal_point=self.ideal_point)
        base = fitness(off)

        # calculate the normalized euclidean distance in the decision space
        X = (self.pop.get("X") - self.F_min) / (self.F_max - self.F_min)
        normalized_distance = squareform(pdist(X))
        np.fill_diagonal(normalized_distance, np.infty)
        L = int(len(self.pop) * self.neighborhood_size)

        rejected = set()

        # selection
        flag_winner = False
        flag_explorer = True
        for i, indv in enumerate(self.sub_pops[j]):
            # the distance from indv to its L-th nearest neighbor in the decision space
            L_nearest_dist = np.sort(normalized_distance[i, :])[L]
            if np.linalg.norm(X[i] - x) < L_nearest_dist:
                flag_explorer = False
                if fitness(indv) > base:
                    rejected.add(i)
                    flag_winner = True
        if flag_winner or flag_explorer:
            self.sub_pops[j].append(off)
        self.sub_pops[j] = [indv for i, indv in enumerate(self.sub_pops[j]) if i not in rejected]

    def select_sparse_solutions(self):
        nds = NonDominatedSorting(method="efficient_non_dominated_sort").do(
            self.pop.get("F"), only_non_dominated_front=True)
        nds_solutions = self.pop[nds]
        selection = DistanceBasedSubsetSelection(based_on="X")
        selected = selection.do(nds_solutions, self.pop_size)
        return nds_solutions[selected]
