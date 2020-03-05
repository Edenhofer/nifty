# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..operators.inversion_enabler import InversionEnabler
from ..operators.sampling_enabler import SamplingEnabler
from ..operators.sandwich_operator import SandwichOperator


def WienerFilterCurvature(R, N, S, iteration_controller=None,
                          iteration_controller_sampling=None):
    """The curvature of the WienerFilterEnergy.

    This operator implements the second derivative of the
    WienerFilterEnergy used in some minimization algorithms or
    for error estimates of the posterior maps. It is the
    inverse of the propagator operator.

    Parameters
    ----------
    R : LinearOperator
        The response operator of the Wiener filter measurement.
    N : EndomorphicOperator
        The noise covariance.
    S : DiagonalOperator
        The prior signal covariance
    iteration_controller : IterationController
        The iteration controller to use during numerical inversion via
        ConjugateGradient.
    iteration_controller_sampling : IterationController
        The iteration controller to use for sampling.
    """
    M = SandwichOperator.make(R, N.inverse)
    if iteration_controller_sampling is not None:
        op = SamplingEnabler(M, S.inverse, iteration_controller_sampling,
                             S.inverse)
    else:
        op = M + S.inverse
    return InversionEnabler(op, iteration_controller, S.inverse)
