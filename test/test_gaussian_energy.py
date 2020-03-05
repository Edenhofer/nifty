# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest

import nifty6 as ift


def _flat_PS(k):
    return np.ones_like(k)


pmp = pytest.mark.parametrize


@pmp('space', [
    ift.GLSpace(15),
    ift.RGSpace(64, distances=.789),
    ift.RGSpace([32, 32], distances=.789)
])
@pmp('nonlinearity', ["tanh", "exp", ""])
@pmp('noise', [1, 1e-2, 1e2])
@pmp('seed', [4, 78, 23])
def test_gaussian_energy(space, nonlinearity, noise, seed):
    np.random.seed(seed)
    dim = len(space.shape)
    hspace = space.get_default_codomain()
    ht = ift.HarmonicTransformOperator(hspace, target=space)
    binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=False)
    pspace = ift.PowerSpace(hspace, binbounds=binbounds)
    Dist = ift.PowerDistributor(target=hspace, power_space=pspace)
    xi0 = ift.Field.from_random(domain=hspace, random_type='normal')

    def pspec(k):
        return 1/(1 + k**2)**dim

    pspec = ift.PS_field(pspace, pspec)
    A = Dist(ift.sqrt(pspec))
    N = ift.ScalingOperator(space, noise)
    n = N.draw_sample()
    R = ift.ScalingOperator(space, 10.)

    def d_model():
        if nonlinearity == "":
            return R(ht(ift.makeOp(A)))
        else:
            tmp = ht(ift.makeOp(A))
            nonlin = getattr(tmp, nonlinearity)()
            return R(nonlin)

    d = d_model()(xi0) + n

    if noise == 1:
        N = None

    energy = ift.GaussianEnergy(d, N)(d_model())
    ift.extra.check_jacobian_consistency(
        energy, xi0, ntries=10, tol=5e-8)
