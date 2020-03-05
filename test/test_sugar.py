# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from numpy.testing import assert_equal

import nifty6 as ift


def test_get_signal_variance():
    space = ift.RGSpace(3)
    hspace = space.get_default_codomain()
    sv = ift.get_signal_variance(lambda x: np.ones_like(x), hspace)
    assert_equal(sv, 3.)

    space = ift.RGSpace(3, distances=1.)
    hspace = space.get_default_codomain()

    def spec2(k):
        t = np.zeros_like(k)
        t[k == 0] = 1.
        return t
    assert_equal(ift.get_signal_variance(spec2, hspace), 1/9.)


def test_exec_time():
    dom = ift.RGSpace(12, harmonic=True)
    op = ift.HarmonicTransformOperator(dom)
    op1 = op.exp()
    lh = ift.GaussianEnergy(domain=op.target) @ op1
    ic = ift.GradientNormController(iteration_limit=2)
    ham = ift.StandardHamiltonian(lh, ic_samp=ic)
    kl = ift.MetricGaussianKL(ift.full(ham.domain, 0.), ham, 1)
    ops = [op, op1, lh, ham, kl]
    for oo in ops:
        for wm in [True, False]:
            ift.exec_time(oo, wm)


def test_calc_pos():
    np.random.seed(42)
    dom = ift.RGSpace(12, harmonic=True)
    op = ift.HarmonicTransformOperator(dom).exp()
    fld = op(0.1*ift.from_random('normal', op.domain))
    pos = ift.calculate_position(op, fld)
    ift.extra.assert_allclose(op(pos), fld, 1e-1, 1e-1)
