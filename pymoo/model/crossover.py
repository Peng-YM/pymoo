import numpy as np
from multimethod import overload, isa

from pymoo.model.population import Population, Individual


class Crossover:
    """
    The crossover combines parents to offsprings. Some crossover are problem specific and use additional information.
    This class must be inherited from to provide a crossover method to an algorithm.
    """

    def __init__(self, n_parents, n_offsprings, prob=0.9):
        self.prob = prob
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings

    def do(self, problem, *args, **kwargs):
        if type(args[0]) is Population:
            pop, parents = args
        else:
            pop = Population.create(*args)
            parents = np.array([np.arange(len(args))])

        if self.n_parents != parents.shape[1]:
            raise ValueError('Exception during crossover: Number of parents differs from defined at crossover.')

        # get the design space matrix form the population and parents
        X = pop.get("X")[parents.T].copy()

        # now apply the crossover probability
        do_crossover = np.random.random(len(parents)) < self.prob

        # execute the crossover
        _X = self._do(problem, X, **kwargs)

        X[:, do_crossover, :] = _X[:, do_crossover, :]

        # flatten the array to become a 2d-array
        X = X.reshape(-1, X.shape[-1])

        # create a population object
        off = pop.new("X", X)

        return off
