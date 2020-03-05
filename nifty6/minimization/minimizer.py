# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..utilities import NiftyMeta


class Minimizer(metaclass=NiftyMeta):
    """A base class used by all minimizers."""

    def __call__(self, energy, preconditioner=None):
        """Performs the minimization of the provided Energy functional.

        Parameters
        ----------
        energy : Energy
           Energy object at the starting point of the iteration

        preconditioner : LinearOperator, optional
           Preconditioner to accelerate the minimization

        Returns
        -------
        Energy : Latest `energy` of the minimization.
        int : exit status of the minimization
            Can be controller.CONVERGED or controller.ERROR
        """
        raise NotImplementedError
