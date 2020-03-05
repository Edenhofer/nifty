# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

import nifty6 as ift
from numpy.testing import assert_, assert_allclose
import pytest

pmp = pytest.mark.parametrize


@pmp('constants', ([], ['a'], ['b'], ['a', 'b']))
@pmp('point_estimates', ([], ['a'], ['b'], ['a', 'b']))
@pmp('mirror_samples', (True, False))
def test_kl(constants, point_estimates, mirror_samples):
    np.random.seed(42)
    dom = ift.RGSpace((12,), (2.12))
    op0 = ift.HarmonicSmoothingOperator(dom, 3)
    op = ift.ducktape(dom, None, 'a')*(op0.ducktape('b'))
    lh = ift.GaussianEnergy(domain=op.target) @ op
    ic = ift.GradientNormController(iteration_limit=5)
    h = ift.StandardHamiltonian(lh, ic_samp=ic)
    mean0 = ift.from_random('normal', h.domain)

    nsamps = 2
    kl = ift.MetricGaussianKL(mean0,
                              h,
                              nsamps,
                              constants=constants,
                              point_estimates=point_estimates,
                              mirror_samples=mirror_samples,
                              napprox=0)
    klpure = ift.MetricGaussianKL(mean0,
                                  h,
                                  nsamps,
                                  mirror_samples=mirror_samples,
                                  napprox=0,
                                  _samples=kl.samples)

    # Test value
    assert_allclose(kl.value, klpure.value)

    # Test gradient
    for kk in h.domain.keys():
        res0 = klpure.gradient[kk].val
        if kk in constants:
            res0 = 0*res0
        res1 = kl.gradient[kk].val
        assert_allclose(res0, res1)

    # Test number of samples
    expected_nsamps = 2*nsamps if mirror_samples else nsamps
    assert_(len(kl.samples) == expected_nsamps)

    # Test point_estimates (after drawing samples)
    for kk in point_estimates:
        for ss in kl.samples:
            ss = ss[kk].val
            assert_allclose(ss, 0*ss)

    # Test constants (after some minimization)
    cg = ift.GradientNormController(iteration_limit=5)
    minimizer = ift.NewtonCG(cg)
    kl, _ = minimizer(kl)
    diff = (mean0 - kl.position).to_dict()
    for kk in constants:
        assert_allclose(diff[kk].val, 0*diff[kk].val)
