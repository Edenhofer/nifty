# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce
from .domain import Domain


class UnstructuredDomain(Domain):
    """A :class:`~nifty6.domains.domain.Domain` subclass for spaces with no
    associated geometry.

    Typically used for data spaces.

    Parameters
    ----------
    shape : tuple of int
        The required shape for an array which can hold the unstructured
        domain's data.
    """

    _needed_for_hash = ["_shape"]

    def __init__(self, shape):
        try:
            self._shape = tuple([int(i) for i in shape])
        except TypeError:
            self._shape = (int(shape), )

    def __repr__(self):
        return "UnstructuredDomain(shape={})".format(self.shape)

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return reduce(lambda x, y: x*y, self.shape)
