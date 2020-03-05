# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from .endomorphic_operator import EndomorphicOperator


class BlockDiagonalOperator(EndomorphicOperator):
    """
    Parameters
    ----------
    domain : MultiDomain
        Domain and target of the operator.
    operators : dict
        Dictionary with subdomain names as keys and :class:`LinearOperator` s
        as items. Any missing item will be treated as unity operator.
    """
    def __init__(self, domain, operators):
        if not isinstance(domain, MultiDomain):
            raise TypeError("MultiDomain expected")
        self._domain = domain
        self._ops = tuple(operators[key] for key in domain.keys())
        self._capability = self._all_ops
        for op in self._ops:
            if op is not None:
                self._capability &= op.capability

    def apply(self, x, mode):
        self._check_input(x, mode)
        val = tuple(op.apply(v, mode=mode) if op is not None else v
                    for op, v in zip(self._ops, x.values()))
        return MultiField(self._domain, val)

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        from ..sugar import from_random
        val = tuple(
            op.draw_sample(from_inverse, dtype)
            if op is not None
            else from_random('normal', self._domain[key], dtype=dtype)
            for op, key in zip(self._ops, self._domain.keys()))
        return MultiField(self._domain, val)

    def _combine_chain(self, op):
        if self._domain != op._domain:
            raise ValueError("domain mismatch")
        res = {key: v1(v2)
               for key, v1, v2 in zip(self._domain.keys(), self._ops, op._ops)}
        return BlockDiagonalOperator(self._domain, res)

    def _combine_sum(self, op, selfneg, opneg):
        from ..operators.sum_operator import SumOperator
        if self._domain != op._domain:
            raise ValueError("domain mismatch")
        res = {key: SumOperator.make([v1, v2], [selfneg, opneg])
               for key, v1, v2 in zip(self._domain.keys(), self._ops, op._ops)}
        return BlockDiagonalOperator(self._domain, res)
