# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..field import Field
from .linear_operator import LinearOperator


class ValueInserter(LinearOperator):
    """Inserts one value into a field which is zero otherwise.

    Parameters
    ----------
    target : Domain, tuple of Domain or DomainTuple
    index : iterable of int
        The index of the target into which the value of the domain shall be
        inserted.
    """

    def __init__(self, target, index):
        self._domain = DomainTuple.scalar_domain()
        self._target = DomainTuple.make(target)
        index = tuple(index)
        if not all([
                isinstance(n, int) and n >= 0 and n < self.target.shape[i]
                for i, n in enumerate(index)
        ]):
            raise TypeError
        if not len(index) == len(self.target.shape):
            raise ValueError
        self._index = index
        self._capability = self.TIMES | self.ADJOINT_TIMES
        # Check whether index is in bounds
        np.empty(self.target.shape)[self._index]

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = np.zeros(self.target.shape, dtype=x.dtype)
            res[self._index] = x
            return Field(self._tgt(mode), res)
        else:
            return Field.scalar(x[self._index])
