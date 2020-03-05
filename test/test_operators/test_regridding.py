# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from numpy.testing import assert_allclose

import nifty6 as ift

from ..common import list2fixture

s = list2fixture([
    ift.RGSpace(8, distances=12.9),
    ift.RGSpace(59, distances=.24, harmonic=True),
    ift.RGSpace([12, 3])
])


def test_value(s):
    Regrid = ift.RegriddingOperator(s, s.shape)
    f = ift.from_random('normal', Regrid.domain)
    assert_allclose(f.val, Regrid(f).val)
