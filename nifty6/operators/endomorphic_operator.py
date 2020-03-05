# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .linear_operator import LinearOperator


class EndomorphicOperator(LinearOperator):
    """Represents a :class:`LinearOperator` which is endomorphic, i.e. one
    which has identical domain and target.
    """
    @property
    def target(self):
        """DomainTuple : returns :attr:`domain`

        Returns `self.domain`, because this is also the target domain
        for endomorphic operators."""
        return self._domain

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        """Generate a zero-mean sample

        Generates a sample from a Gaussian distribution with zero mean and
        covariance given by the operator.

        Parameters
        ----------
        from_inverse : bool (default : False)
            if True, the sample is drawn from the inverse of the operator
        dtype : numpy datatype (default : numpy.float64)
            the data type to be used for the sample

        Returns
        -------
        Field
            A sample from the Gaussian of given covariance.
        """
        raise NotImplementedError

    def _dom(self, mode):
        return self._domain

    def _tgt(self, mode):
        return self._domain

    def _check_input(self, x, mode):
        self._check_mode(mode)
        if self.domain != x.domain:
            raise ValueError("The operator's and field's domains don't match.")
