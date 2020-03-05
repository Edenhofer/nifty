# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from .energy import Energy


class QuadraticEnergy(Energy):
    """The Energy for a quadratic form.
    The most important aspect of this energy is that its metric must be
    position-independent.
    """

    def __init__(self, position, A, b, _grad=None):
        super(QuadraticEnergy, self).__init__(position=position)
        self._A = A
        self._b = b
        if _grad is not None:
            self._grad = _grad
            Ax = _grad if b is None else _grad + b
        else:
            Ax = self._A(self._position)
            self._grad = Ax if b is None else Ax - b
        self._value = 0.5*self._position.vdot(Ax)
        if b is not None:
            self._value -= b.vdot(self._position)

    def at(self, position):
        return QuadraticEnergy(position, self._A, self._b)

    def at_with_grad(self, position, grad):
        """Specialized version of `at`, taking also a gradient.

        This custom method is meant for use within :class:ConjugateGradient`
        minimizers, which already have the gradient available. It saves time
        by not recomputing it.

        Parameters
        ----------
        position : Field
            Location in parameter space for the new Energy object.
        grad : Field
            Energy gradient at the new position.

        Returns
        -------
        Energy
            Energy object at new position.
        """
        return QuadraticEnergy(position, self._A, self._b, grad)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._grad

    @property
    def metric(self):
        return self._A

    def apply_metric(self, x):
        return self._A(x)
