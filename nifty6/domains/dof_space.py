# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .structured_domain import StructuredDomain


class DOFSpace(StructuredDomain):
    """Generic degree-of-freedom space. It is defined as the domain of some
    DOFDistributor.
    Its entries represent the underlying degrees of freedom of some other
    space, according to the dofdex.

    Parameters
    ----------
    dof_weights: 1-D numpy array
        A numpy array containing the multiplicity of each individual degree of
        freedom.
    """

    _needed_for_hash = ["_dvol"]

    def __init__(self, dof_weights):
        self._dvol = tuple(dof_weights)

    @property
    def harmonic(self):
        return False

    @property
    def shape(self):
        return (len(self._dvol),)

    @property
    def size(self):
        return len(self._dvol)

    @property
    def scalar_dvol(self):
        return None

    @property
    def dvol(self):
        return np.array(self._dvol)

    def __repr__(self):
        return 'this is a dof space'
