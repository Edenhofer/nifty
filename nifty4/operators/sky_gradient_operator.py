from ..multi import MultiDomain, MultiField
from ..sugar import full
from .linear_operator import LinearOperator


class MultiSkyGradientOperator(LinearOperator):
    def __init__(self, gradients, domain, target):
        super(MultiSkyGradientOperator, self).__init__()
        self._gradients = gradients
        gradients_domain = MultiField(self._gradients).domain
        self._domain = MultiDomain.make(domain)

        # Check compatibility
        # assert gradients_domain.items() <= self.domain.items()
        # FIXME This is a python2 hack!
        assert all(item in self.domain.items() for item in gradients_domain.items())

        self._target = target
        for grad in gradients.values():
            if self._target != grad.target:
                raise TypeError(
                    'All gradients have to have the same target domain')

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    @property
    def gradients(self):
        return self._gradients

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = None
            for key, op in self._gradients.items():
                if res is None:
                    res = op(x[key])
                else:
                    res += op(x[key])
            # Needed if gradients == {}
            if res is None:
                res = full(self.target, 0.)
            assert res.domain == self.target
        else:
            grad_keys = self._gradients.keys()
            res = {}
            for dd in self.domain:
                if dd in grad_keys:
                    res[dd] = self._gradients[dd].adjoint_times(x)
                else:
                    res[dd] = full(self.domain[dd], 0.)
            res = MultiField(res)
            assert res.domain == self.domain
        return res