# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .structured_domain import StructuredDomain


class HPSpace(StructuredDomain):
    """Represents 2-sphere with HEALPix discretization.

    Its harmonic partner domain is the
    :class:`~nifty6.domains.lm_space.LMSpace`.

    Parameters
    ----------
    nside : int
        The corresponding HEALPix Nside parameter. Must be a positive integer
        and typically is a power of 2.
    """

    _needed_for_hash = ["_nside"]

    def __init__(self, nside):
        self._nside = int(nside)
        if self._nside < 1:
            raise ValueError("nside must be >=1.")

    def __repr__(self):
        return "HPSpace(nside={})".format(self.nside)

    @property
    def harmonic(self):
        return False

    @property
    def shape(self):
        return (self.size,)

    @property
    def size(self):
        return np.int(12 * self.nside * self.nside)

    @property
    def scalar_dvol(self):
        return np.pi / (3*self._nside*self._nside)

    @property
    def nside(self):
        """int : HEALPix Nside parameter of this domain"""
        return self._nside

    def get_default_codomain(self):
        """Returns a :class:`LMSpace` object, which is capable of storing a
        fairly accurate representation of data residing on `self`

        Returns
        -------
        LMSpace
            The partner domain

        Notes
        -----
        The `lmax` and `mmax` parameters of the returned :class:`LMSpace` are
        set to `2*self.nside`.
        """
        from ..domains.lm_space import LMSpace
        return LMSpace(lmax=2*self.nside)

    def check_codomain(self, codomain):
        """Raises `TypeError` if `codomain` is not a matching partner domain
        for `self`.

        Notes
        -----
        This function only checks whether `codomain` is of type
        :class:`LMSpace`.
        """
        from ..domains.lm_space import LMSpace
        if not isinstance(codomain, LMSpace):
            raise TypeError("codomain must be a LMSpace.")
