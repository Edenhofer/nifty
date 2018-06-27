from nifty5 import Energy, InversionEnabler, SamplingEnabler, Variable, memo
from nifty5.library import UnitLogGauss


class Hamiltonian(Energy):
    def __init__(self, lh, iteration_controller,
                 iteration_controller_sampling=None):
        """
        lh: Likelihood (energy object)
        prior:
        """
        super(Hamiltonian, self).__init__(lh.position)
        self._lh = lh
        self._ic = iteration_controller
        if iteration_controller_sampling is None:
            self._ic_samp = iteration_controller
        else:
            self._ic_samp = iteration_controller_sampling
        self._prior = UnitLogGauss(Variable(self.position))
        self._precond = self._prior.curvature

    def at(self, position):
        return self.__class__(self._lh.at(position), self._ic, self._ic_samp)

    @property
    @memo
    def value(self):
        return self._lh.value + self._prior.value

    @property
    @memo
    def gradient(self):
        return self._lh.gradient + self._prior.gradient

    @property
    @memo
    def curvature(self):
        prior_curv = self._prior.curvature
        c = SamplingEnabler(self._lh.curvature, prior_curv.inverse,
                            self._ic_samp, prior_curv.inverse)
        return InversionEnabler(c, self._ic, self._precond)

    def __str__(self):
        res = 'Likelihood:\t{:.2E}\n'.format(self._lh.value)
        res += 'Prior:\t\t{:.2E}'.format(self._prior.value)
        return res
