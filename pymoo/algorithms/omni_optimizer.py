import numpy as np
from scipy.spatial.distance import pdist, squareform

from pymoo.algorithms.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.model.selection import Selection
from pymoo.util.dominator import Dominator


class OmniOptimizer(NSGA2):
    def __init__(self,
                 **kwargs):
        super().__init__(
            selection=ParentSelection(),
            survival=RankAndCrowdingSurvival(
                cdist_func=special_crowding_distance
            ),
            **kwargs
        )


def special_crowding_distance(pop_front):
    N = len(pop_front)
    if N == 1:
        return np.array([0])

    normalized_var = normalize(pop_front.get("X"))
    normalized_obj = normalize(pop_front.get("F"))

    # initialize all crowding distance to zero
    crowd_dist_var = np.zeros(N)
    crowd_dist_obj = np.zeros(N)

    # objective space crowding
    n_obj = normalized_obj.shape[1]
    for j in range(n_obj):
        ranks, I = rank_sort(normalized_obj[:, j])
        for i in range(N):
            if ranks[i] == 0 or ranks[i] == N - 1:
                crowd_dist_obj[i] = np.infty
            else:
                crowd_dist_obj[i] += (normalized_obj[I[ranks[i] + 1], j] - normalized_obj[I[ranks[i]], j])
    crowd_dist_obj /= n_obj

    # decision space crowding
    n_var = normalized_var.shape[1]
    for k in range(n_var):
        ranks, I = rank_sort(normalized_var[:, k])
        for i in range(N):
            if ranks[i] == 0:
                crowd_dist_var[i] += 2 * (normalized_var[I[1], k] - normalized_var[I[0], k])
            elif ranks[i] == N - 1:
                crowd_dist_var[i] += 2 * (normalized_var[I[N - 1], k] - normalized_var[I[N - 2], k])
            else:
                crowd_dist_var[i] += (normalized_var[I[ranks[i] + 1], k] - normalized_var[I[ranks[i]], k])
    crowd_dist_var /= n_var

    # aggregate the final crowding distance
    avg_crowd_dist_var = np.mean(crowd_dist_var)
    avg_crowd_dist_obj = np.mean(crowd_dist_obj)

    crowd_dist = np.zeros(N)
    for i in range(N):
        if crowd_dist_var[i] > avg_crowd_dist_var or crowd_dist_obj[i] > avg_crowd_dist_obj:
            crowd_dist[i] = max(crowd_dist_var[i], crowd_dist_obj[i])
        else:
            crowd_dist[i] = min(crowd_dist_var[i], crowd_dist_obj[i])

    return crowd_dist


class ParentSelection(Selection):
    def _do(self, pop, n_select, n_parents, **kwargs):
        assert n_parents == 2

        N = len(pop)
        Rt = np.hstack((np.random.permutation(N), np.random.permutation(N)))
        X = normalize(pop.get("X"))
        # pre-calculate the pairwise distance in the decision space
        D = squareform(pdist(X))
        mating_pool = np.zeros((n_select, 2), dtype=int)
        for i in range(n_select):
            # first selection
            player1, player2 = Rt[choose_nearest(D)]
            parent1 = tournament(pop, player1, player2)
            # second selection
            player1, player2 = Rt[choose_nearest(D)]
            parent2 = tournament(pop, player1, player2)
            mating_pool[i] = [parent1, parent2]
        return mating_pool


def choose_nearest(D):
    N = D.shape[0]
    # randomly pop a solution
    a = np.random.randint(N)
    # find the closest neighbor of a in the decision space
    remains = np.arange(N) != a
    b = np.argmin(D[a, remains])
    return np.array([a, b])


def tournament(pop, ai, bi):
    a, b = pop[ai], pop[bi]
    relation = Dominator.get_relation(a.F, b.F, a.CV, b.CV)
    if relation == 1:
        return ai
    elif relation == -1:
        return bi
    else:
        candidates = [ai, bi]
        return candidates[np.random.choice(1)]


def normalize(V):
    v_min = np.min(V, axis=0)
    v_max = np.max(V, axis=0)
    return (V - v_min) / (v_max - v_min)


def rank_sort(array):
    I = np.argsort(array)
    ranks = np.empty_like(I)
    ranks[I] = np.arange(len(array))
    return ranks, I
