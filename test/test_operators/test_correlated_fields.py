# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import pytest
from numpy.random import seed
from numpy.testing import assert_allclose

import nifty6 as ift


@pytest.mark.parametrize('sspace', [
    ift.RGSpace(4),
    ift.RGSpace((4, 4), (0.123, 0.4)),
    ift.HPSpace(8),
    ift.GLSpace(4)
])
@pytest.mark.parametrize('rseed', [13, 2])
@pytest.mark.parametrize('Astds', [[1., 3.], [0.2, 1.4]])
@pytest.mark.parametrize('offset_std', [1., 10.])
@pytest.mark.parametrize('N', [0,2])
def testAmplitudesConsistency(rseed, sspace, Astds, offset_std, N):
    def stats(op, samples):
        sc = ift.StatCalculator()
        for s in samples:
            sc.add(op(s.extract(op.domain)))
        return sc.mean.val, sc.var.sqrt().val

    seed(rseed)
    nsam = 100

    fsspace = ift.RGSpace((12,), (0.4,))
    if N==2:
        dofdex1 = [0,0]
        dofdex2 = [1,0]
        dofdex3 = [1,1]
    else:
        dofdex1, dofdex2, dofdex3 = None, None, None

    fa = ift.CorrelatedFieldMaker.make(offset_std, 1E-8, '', N, dofdex1)
    fa.add_fluctuations(sspace, Astds[0], 1E-8, 1.1, 2., 2.1, .5, -2, 1.,
                        'spatial', dofdex = dofdex2)
    fa.add_fluctuations(fsspace, Astds[1], 1E-8, 3.1, 1., .5, .1, -4, 1.,
                        'freq', dofdex = dofdex3)
    op = fa.finalize()

    samples = [ift.from_random('normal', op.domain) for _ in range(nsam)]
    tot_flm, _ = stats(fa.total_fluctuation, samples)
    offset_amp_std, _ = stats(fa.amplitude_total_offset, samples)
    intergated_fluct_std0, _ = stats(fa.average_fluctuation(0), samples)
    intergated_fluct_std1, _ = stats(fa.average_fluctuation(1), samples)

    slice_fluct_std0, _ = stats(fa.slice_fluctuation(0), samples)
    slice_fluct_std1, _ = stats(fa.slice_fluctuation(1), samples)

    sams = [op(s) for s in samples]
    fluct_total = fa.total_fluctuation_realized(sams)
    fluct_space = fa.average_fluctuation_realized(sams, 0)
    fluct_freq = fa.average_fluctuation_realized(sams, 1)
    zm_std_mean = fa.offset_amplitude_realized(sams)
    sl_fluct_space = fa.slice_fluctuation_realized(sams, 0)
    sl_fluct_freq = fa.slice_fluctuation_realized(sams, 1)

    assert_allclose(offset_amp_std, zm_std_mean, rtol=0.5)
    assert_allclose(intergated_fluct_std0, fluct_space, rtol=0.5)
    assert_allclose(intergated_fluct_std1, fluct_freq, rtol=0.5)
    assert_allclose(tot_flm, fluct_total, rtol=0.5)
    assert_allclose(slice_fluct_std0, sl_fluct_space, rtol=0.5)
    assert_allclose(slice_fluct_std1, sl_fluct_freq, rtol=0.5)

    
    fa = ift.CorrelatedFieldMaker.make(offset_std, .1, '', N, dofdex1)
    fa.add_fluctuations(fsspace, Astds[1], 1., 3.1, 1., .5, .1, -4, 1., 'freq', dofdex = dofdex3)
    m = 3.
    x = fa.moment_slice_to_average(m)
    fa.add_fluctuations(sspace, x, 1.5, 1.1, 2., 2.1, .5, -2, 1., 'spatial', 0, dofdex = dofdex2)
    op = fa.finalize()
    em, estd = stats(fa.slice_fluctuation(0), samples)

    assert_allclose(m, em, rtol=0.5)

    assert op.target[-2] == sspace
    assert op.target[-1] == fsspace
