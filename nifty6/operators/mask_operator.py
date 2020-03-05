# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from .linear_operator import LinearOperator


class MaskOperator(LinearOperator):
    """Implementation of a mask response

    Takes a field, applies flags and returns the values of the field in a
    :class:`UnstructuredDomain`.

    Parameters
    ----------
    flags : Field
        Is converted to boolean. Where True, the input field is flagged.
    """
    def __init__(self, flags):
        if not isinstance(flags, Field):
            raise TypeError
        self._domain = DomainTuple.make(flags.domain)
        self._flags = np.logical_not(flags.val)
        self._target = DomainTuple.make(UnstructuredDomain(self._flags.sum()))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = x[self._flags]
            return Field(self.target, res)
        res = np.empty(self.domain.shape, x.dtype)
        res[self._flags] = x
        res[~self._flags] = 0
        return Field(self.domain, res)
