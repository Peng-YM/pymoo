from abc import ABC

from pymoo.model.problem import Problem


class MMF(Problem, ABC):
    """
    The MMF test suite is a multi-modal multi-objective optimization test suites proposed in [1]. It has been used in the
    special session of the CEC 2019.

    References
    ----------
    [1] Yue, C., Qu, B., Liang, J. "A Multiobjective Particle Swarm Optimizer Using Ring Topology for Solving \
    Multimodal Multiobjective Problems"
    """

    def _calc_pareto_front(self, n_pareto_points=10000):
        PS = self._calc_pareto_set(n_pareto_points)
        return self.evaluate(PS, return_values_of=["F"])
