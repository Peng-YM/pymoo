from numpy import exp, ones
from numpy.linalg import inv
from scipy.spatial.distance import pdist

from pymoo.model.indicator import Indicator


class SolowPolaskyIndicator(Indicator):
    def __init__(self, theta=1):
        self.theta = theta

    def _calc(self, X):
        n, _ = X.shape
        M = exp(-self.theta * pdist(X))
        e = ones(n)
        return e @ inv(M) @ e.T
