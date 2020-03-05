# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .. import utilities
from ..domain_tuple import DomainTuple
from ..field import Field
from .linear_operator import LinearOperator


class ContractionOperator(LinearOperator):
    """A :class:`LinearOperator` which sums up fields into the direction of
    subspaces.

    This Operator sums up a field with is defined on a :class:`DomainTuple`
    to a :class:`DomainTuple` which is a subset of the former.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
    spaces : None, int or tuple of int
        The elements of "domain" which are contracted.
        If `None`, everything is contracted
    weight : int, default=0
        If nonzero, the fields defined on self.domain are weighted with the
        specified power along the submdomains which are contracted.
    """

    def __init__(self, domain, spaces, weight=0):
        self._domain = DomainTuple.make(domain)
        self._spaces = utilities.parse_spaces(spaces, len(self._domain))
        self._target = [
            dom for i, dom in enumerate(self._domain) if i not in self._spaces
        ]
        self._target = DomainTuple.make(self._target)
        self._weight = weight
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.ADJOINT_TIMES:
            ldat = x.val
            shp = []
            for i, dom in enumerate(self._domain):
                tmp = dom.shape
                shp += tmp if i not in self._spaces else (1,)*len(dom.shape)
            ldat = np.broadcast_to(ldat.reshape(shp), self._domain.shape)
            res = Field(self._domain, ldat)
            if self._weight != 0:
                res = res.weight(self._weight, spaces=self._spaces)
            return res
        else:
            if self._weight != 0:
                x = x.weight(self._weight, spaces=self._spaces)
            res = x.sum(self._spaces)
            return res if isinstance(res, Field) else Field.scalar(res)
