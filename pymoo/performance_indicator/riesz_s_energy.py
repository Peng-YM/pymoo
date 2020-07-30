import numpy as np
from scipy.spatial.distance import pdist, squareform


class RieszSEnergyIndicator:
    def __init__(self, s=None):
        self.s = s

    def calc(self, A):
        n, m = A.shape
        s = self.s or m ** 2
        I, J = np.triu_indices(len(A), 1)
        print(I, J)
        D = squareform(pdist(A))[I, J] ** s
        D = D[D != 0]  # handle division-by-zero errors!
        return np.sum(1 / D) / n


if __name__ == '__main__':
    X = np.array([
        [0, 0], [0, 1], [1, 1], [1, 0]
    ])
    indicator = RieszSEnergyIndicator()

    print("Riesz-s Energy is: ", indicator.calc(X))
