import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from pymoo.model.subset_selection import SubsetSelection, normalize
from pymoo.subset_selection.distance_based_subset_selection import DistanceBasedSubsetSelection


class DBCANClustering:
    def __init__(self, epsilon, min_samples):
        self.epsilon = epsilon
        self.min_samples = min_samples

    def do(self, V):
        model = DBSCAN(eps=self.epsilon, min_samples=self.min_samples)
        return model.fit(V).labels_


class TwoStageSubsetSelection(SubsetSelection):
    def __init__(self, clustering, delta_max):
        self.clustering = clustering
        self.delta_max = delta_max

    def _do(self, pop, n_select, **kwargs):
        X = normalize(pop.get("X"))
        F = normalize(pop.get("F"))
        selected = np.full(len(pop), False)
        # select extreme solutions
        I = self.select_extreme_solutions(X, F)
        selected[I] = True
        # select other solutions
        cnt = len(I)
        while cnt < n_select:
            # stage 1: DSS in objective space
            remain = np.where(~selected)[0]  # indices of unselected solutions
            # calculate distance from each unselected solution to nearest selected solution.
            PD = cdist(F[~selected], F[selected])
            ND = np.min(PD, axis=1)
            # maximize such distance
            j = np.argmax(ND, axis=0)
            i = remain[j]
            # stage 2:  select equivalent solutions of i
            I = self.select_equivalent_solutions(X, F, i)
            selected[I] = True
            cnt = cnt + len(I)
        return selected

    def select_extreme_solutions(self, X, F):
        M = F.shape[1]
        selected = None
        for m in range(M):
            i = np.argmin(F[:, m])
            I = self.select_equivalent_solutions(X, F, i)
            if selected is None:
                selected = I
            else:
                selected = np.hstack((selected, I))
        return selected

    def select_equivalent_solutions(self, X, F, i):
        # find candidate equivalent solutions for solution i
        D = cdist(F, np.asarray([F[i]]))
        candidate_indices = np.where(D < self.delta_max)[0]
        # clustering according to decision variables
        cluster_labels = self.clustering.do(X[candidate_indices])
        n_clusters = np.max(cluster_labels) + 1
        # for each cluster, select the solution closest to solution i in the objective space
        selected = np.zeros(n_clusters, dtype=np.int)
        for c in range(n_clusters):
            element_indices = candidate_indices[cluster_labels == c]
            D = cdist(F[element_indices], np.asarray([F[i]]))
            j = np.argmin(D)
            selected[c] = element_indices[j]
        return selected


class ModifiedTwoStageSubsetSelection(TwoStageSubsetSelection):
    def __init__(self, clustering, delta_max, objective_selector=DistanceBasedSubsetSelection()):
        self.objective_selector = objective_selector
        super().__init__(clustering, delta_max)

    def _do(self, pop, n_select, **kwargs):
        # normalize
        X = normalize(pop.get("X"))
        F = normalize(pop.get("F"))
        # step 1: select sparse solutions in the decision space
        selected_idx = self.objective_selector.do(pop, n_select)
        # step 2: select equivalent solutions for each selected solution
        # each solution set stores indices of equivalent solutions
        equivalent_solution_sets = [None for _ in range(len(selected_idx))]
        for i, index in enumerate(selected_idx):
            equivalent_solution_sets[i] = self.select_equivalent_solutions(X, F, index)
        # step 3: select one solution in each equivalent solution set
        # sort the equivalent solution sets based on cardinality.
        selected = np.full(len(pop), False)
        equivalent_solution_sets.sort(key=lambda s: len(s))
        for set_ in equivalent_solution_sets:
            if ~np.any(selected):
                # if no solution has been select randomly select a solution from this solution set
                selected[np.random.choice(set_)] = True
            else:
                # otherwise try to select a solution in this solution set which has maximum distance
                # to its neighborhood solutions in the decision space.
                D = cdist(X[set_], X[selected])
                j = np.argmax(np.min(D, axis=1), axis=0)
                i = set_[j]
                selected[i] = True
        return selected
