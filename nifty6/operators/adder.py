# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..field import Field
from ..multi_field import MultiField
from .operator import Operator


class Adder(Operator):
    """Adds a fixed field.

    Parameters
    ----------
    field : Field or MultiField
        The field by which the input is shifted.
    """
    def __init__(self, field, neg=False):
        if not isinstance(field, (Field, MultiField)):
            raise TypeError
        self._field = field
        self._domain = self._target = field.domain
        self._neg = bool(neg)

    def apply(self, x):
        self._check_input(x)
        if self._neg:
            return x - self._field
        return x + self._field
