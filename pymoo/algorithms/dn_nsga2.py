import numpy as np
from scipy.spatial.distance import pdist, squareform

from pymoo.algorithms.nsga2 import NSGA2, RankAndCrowdingSurvival, calc_crowding_distance
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.util.dominator import Dominator


class DN_NSGA2(NSGA2):
    def __init__(self,
                 crowding_factor,
                 **kwargs):
        """
        Decision space based niching NSGAII (DN-NSGAII)

        Parameters
        ----------
        crowding_factor: the crowding factor
        """
        super().__init__(
            selection=CrowdingFactorSelection(crowding_factor),
            survival=RankAndCrowdingSurvival(cdist_func=lambda pop: calc_crowding_distance(pop.get("X"))),
            **kwargs
        )


class CrowdingFactorSelection(RandomSelection):
    def __init__(self, crowding_factor):
        super().__init__()
        self.cf = crowding_factor

    def _do(self, pop, n_select, n_parents, **kwargs):
        N = len(pop)
        X = pop.get("X").astype(np.float, copy=False)

        # the mating pool
        pool = []
        # pre-calculate pairwise distance
        D = squareform(pdist(X))
        cnt = 0
        while cnt < n_select:
            # randomly select a solution from the population
            a = np.random.randint(N)
            # select cf solutions from the left solutions
            B = np.random.choice([i for i in range(N) if i != a], self.cf)
            # find the closest neighbor in B with respect to the decision variables
            b = np.argmin(D[a, B])
            # insert the superior one to the mating pool
            relation = Dominator.get_relation(pop[a].F, pop[b].F)
            if relation == 1:
                selected = a
            elif relation == -1:
                selected = b
            else:
                selected = np.random.choice([a, b])
            pool.append(selected)

            cnt += 1

        # apply random selection to the mating pool
        return super()._do(pop[pool], n_select, n_parents)
