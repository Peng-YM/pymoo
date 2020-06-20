import numpy as np
from scipy.spatial.distance import cdist

from pymoo.factory import get_reference_directions
from pymoo.model.subset_selection import normalize, SubsetSelection


class WeightVectorBasedSubsetSelection(SubsetSelection):
    """
    A subset selection method based on weight vectors which proposed in [1]. For each weight vector, a solution with the
    minimum angle to it will be selected.

    References
    ----------
    [1] H. K. Singh, K. S. Bhattacharjee, and T. Ray, “Distance-based subset selection for benchmarking in evolutionary multi/many-objective optimization,” IEEE Transactions on Evolutionary Computation, vol. 23, no. 5, pp. 904–912, October 2019.
    """

    def _do(self, n_select, ref_dirs=None):
        """
        Parameters
        ----------
        n_select:
            number of solutions to be selected.
        ref_dirs:
            weight vectors.
        Returns
        -------
        The indices of the selected solutions.
        """
        if ref_dirs is not None and len(ref_dirs) > n_select:
            raise ValueError("The number of reference vectors should smaller or equal to the number of solutions to "
                             "be selected!")
        V = self.pop.get("F")
        # normalize objective value
        V = normalize(V)
        # if no weight vectors are provided, generate reference vectors using Riesz s-Energy method to generate them.
        if ref_dirs is None:
            ref_dirs = get_reference_directions('energy', V.shape[1], n_select)
        # for each weight vector, select one solution that has minimum angle to it. Here we use the cosine distance.
        selected = np.full(len(self.pop), False)
        PD = cdist(V, ref_dirs, metric='cosine')
        I = np.argmin(PD, axis=0)
        selected[I] = True

        return selected
