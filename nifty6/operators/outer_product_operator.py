# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..field import Field
from .linear_operator import LinearOperator


class OuterProduct(LinearOperator):
    """Performs the point-wise outer product of two fields.

    Parameters
    ---------
    field: Field,
    domain: DomainTuple, the domain of the input field
    ---------
    """
    def __init__(self, field, domain):
        self._domain = domain
        self._field = field
        self._target = DomainTuple.make(
            tuple(sub_d for sub_d in field.domain._dom + domain._dom))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return Field(
                self._target, np.multiply.outer(
                    self._field.val, x.val))
        axes = len(self._field.shape)
        return Field(
            self._domain, np.tensordot(self._field.val, x.val, axes))
