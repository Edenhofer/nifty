# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .linear_operator import LinearOperator


class OperatorAdapter(LinearOperator):
    """Class representing the inverse and/or adjoint of another operator.

    Objects of this class are created internally by `LinearOperator` whenever
    the inverse and/or adjoint of an already existing operator object is
    requested via the `LinearOperator` attributes `inverse`, `adjoint` or
    `_flip_modes()`.

    Users should never have to create instances of this class directly.

    Parameters
    ----------
    op : LinearOperator
        The operator on which the adapter will act
    op_transform : int
        1) adjoint
        2) inverse
        3) adjoint inverse
    """

    def __init__(self, op, op_transform):
        self._op = op
        self._trafo = int(op_transform)
        if self._trafo < 1 or self._trafo > 3:
            raise ValueError("invalid operator transformation")
        self._domain = self._op._dom(1 << self._trafo)
        self._target = self._op._tgt(1 << self._trafo)
        self._capability = self._capTable[self._trafo][self._op.capability]

    def _flip_modes(self, trafo):
        newtrafo = trafo ^ self._trafo
        return self._op if newtrafo == 0 \
            else OperatorAdapter(self._op, newtrafo)

    def apply(self, x, mode):
        return self._op.apply(x,
                              self._modeTable[self._trafo][self._ilog[mode]])

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        if self._trafo & self.INVERSE_BIT:
            return self._op.draw_sample(not from_inverse, dtype)
        return self._op.draw_sample(from_inverse, dtype)

    def __repr__(self):
        from ..utilities import indent
        mode = ["adjoint", "inverse", "adjoint inverse"][self._trafo-1]
        res = "OperatorAdapter: {}\n".format(mode)
        return res + indent(self._op.__repr__())
