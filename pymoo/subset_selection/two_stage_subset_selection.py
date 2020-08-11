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
        # normalization
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
            # stage 2: select equivalent solutions of i
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
        candidate_indices = np.where(D <= self.delta_max)[0]
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

    def _do(self, pop, n_select, max_equivalent_solutions=1, **kwargs):
        selected = np.full(len(pop), False)

        # normalize
        X = normalize(pop.get("X"))
        F = normalize(pop.get("F"))

        n_var, n_obj = X.shape[1], F.shape[1]

        # step 1: select sparse solutions in the objective space
        selected_idx = self.objective_selector.do(pop, n_select)

        # step 2: select equivalent solutions for each selected solution yi
        equivalent_solutions_indices = []
        candidates = []

        for i, index in enumerate(selected_idx):
            # indices of the equivalent solutions corresponding to yi
            I = self.select_equivalent_solutions(X, F, index)
            equivalent_solutions_indices.append(set(I))
            candidates.extend(I)

        parents = {}  # a map for finding the set of equivalent solutions
        candidates = np.array(candidates)

        for i, set_ in enumerate(equivalent_solutions_indices):
            for idx in set_:
                parents[idx] = i

        n_selected_equivalent_solutions = [0 for _ in range(len(equivalent_solutions_indices))]

        # step 3: keep one solution in each equivalent solution set
        cnt = 0
        while cnt < n_select and len(candidates) > 0:
            # (1) randomly select a boundary solution in the decision space as an initial solution
            if not np.any(selected):
                idx = np.argmin(X[candidates, np.random.randint(0, n_var)])
                idx = candidates[idx]
            # (2) DSS in the decision space
            else:
                PD = cdist(X[candidates], X[selected])
                ND = np.min(PD, axis=1)  # distance to nearest neighbor in selected set
                # maximize such distance
                j = np.argmax(ND, axis=0)
                idx = candidates[j]

            # select the solution
            selected[idx] = True
            cnt += 1
            n_selected_equivalent_solutions[parents[idx]] += 1
            # exclude its equivalent solutions from candidates
            if n_selected_equivalent_solutions[parents[idx]] < max_equivalent_solutions:
                candidates = candidates[candidates != idx]
            else:
                set_ = equivalent_solutions_indices[parents[idx]]
                candidates = candidates[[i for i, c in enumerate(candidates) if c not in set_]]

        return selected
