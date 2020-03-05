# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..utilities import NiftyMeta


class Energy(metaclass=NiftyMeta):
    """Provides the functional used by minimization schemes.

    The Energy object is an implementation of a scalar function including its
    gradient and metric at some position.

    Parameters
    ----------
    position : Field
        The input parameter of the scalar function.

    Notes
    -----
    An instance of the Energy class is defined at a certain location. If one
    is interested in the value, gradient or metric of the abstract energy
    functional one has to 'jump' to the new position using the `at` method.
    This method returns a new energy instance residing at the new position. By
    this approach, intermediate results from computing e.g. the gradient can
    safely be reused for e.g. the value or the metric.

    Memorizing the evaluations of some quantities minimizes the computational
    effort for multiple calls.
    """

    def __init__(self, position):
        self._position = position
        self._gradnorm = None

    def at(self, position):
        """Returns a new Energy object, initialized at `position`.

        Parameters
        ----------
        position : Field
            Location in parameter space for the new Energy object.

        Returns
        -------
        Energy
            Energy object at new position.
        """
        return self.__class__(position)

    @property
    def position(self):
        """
        Field : selected location in parameter space.

        The Field location in parameter space where value, gradient and
        metric are evaluated.
        """
        return self._position

    @property
    def value(self):
        """
        float : value of the functional.

            The value of the energy functional at given `position`.
        """
        raise NotImplementedError

    @property
    def gradient(self):
        """
        Field : The gradient at given `position`.
        """
        raise NotImplementedError

    @property
    def gradient_norm(self):
        """
        float : L2-norm of the gradient at given `position`.
        """
        if self._gradnorm is None:
            self._gradnorm = self.gradient.norm()
        return self._gradnorm

    @property
    def metric(self):
        """
        LinearOperator : implicitly defined metric.
            A positive semi-definite operator or function describing the
            metric of the potential at the given `position`.
        """
        raise NotImplementedError

    def apply_metric(self, x):
        """
        Parameters
        ----------
        x: Field or MultiField
            Argument for the metric operator

        Returns
        -------
        Field or MultiField:
            Output of the metric operator
        """
        raise NotImplementedError

    def longest_step(self, dir):
        """Returns the longest allowed step size along `dir`

        Parameters
        ----------
        dir : Field
            the search direction

        Returns
        -------
        float or None
            the longest allowed step when starting from `self.position` along
            `dir`. If None, the step size is not limited.
        """
        return None
