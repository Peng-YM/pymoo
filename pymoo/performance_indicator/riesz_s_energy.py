from numpy import sum
from scipy.spatial.distance import pdist, squareform


class RieszSEnergyIndicator:
    def __init__(self, s=None):
        self.s = s

    def calc(self, A):
        n, m = A.shape
        s = self.s or m - 1
        I, J = np.triu_indices(len(A), 1)
        D = squareform(pdist(A))[I, J] ** s
        return 2 * sum(1 / D)


if __name__ == '__main__':
    import numpy as np

    X = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    indicator = RieszSEnergyIndicator()

    assert indicator.calc(X) == (4 + np.sqrt(2)) * 2

    print("Riesz-s Energy is: ", indicator.calc(X))
