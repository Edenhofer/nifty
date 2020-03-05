# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..field import Field
from .linear_operator import LinearOperator


class DomainTupleFieldInserter(LinearOperator):
    """Writes the content of a :class:`Field` into one slice of a
    :class:`DomainTuple`.

    Parameters
    ----------
    target : Domain, tuple of Domain or DomainTuple
    space : int
       The index of the sub-domain which is inserted.
    index : tuple
        Slice in new sub-domain in which the input field shall be written into.
    """

    def __init__(self, target, space, pos):
        if not space <= len(target) or space < 0:
            raise ValueError
        self._target = DomainTuple.make(target)
        dom = list(self.target)
        dom.pop(space)
        self._domain = DomainTuple.make(dom)
        self._capability = self.TIMES | self.ADJOINT_TIMES

        new_space = target[space]
        nshp = new_space.shape
        fst_dims = sum(len(dd.shape) for dd in self.target[:space])

        if len(pos) != len(nshp):
            raise ValueError("shape mismatch between new_space and position")
        for s, p in zip(nshp, pos):
            if p < 0 or p >= s:
                raise ValueError("bad position value")

        self._slc = (slice(None),)*fst_dims + pos

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = np.zeros(self.target.shape, dtype=x.dtype)
            res[self._slc] = x.val
            return Field(self.target, res)
        else:
            return Field(self.domain, x.val[self._slc])
