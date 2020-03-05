# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from numpy.testing import assert_

import nifty6 as ift


@pytest.mark.parametrize('sp', [
    ift.RGSpace(4),
    ift.PowerSpace(ift.RGSpace((4, 4), harmonic=True)),
    ift.LMSpace(5),
    ift.HPSpace(4),
    ift.GLSpace(4)
])
@pytest.mark.parametrize('seed', [13, 2])
def test_value_inserter(sp, seed):
    np.random.seed(seed)
    ind = tuple([np.random.randint(0, ss - 1) for ss in sp.shape])
    op = ift.ValueInserter(sp, ind)
    f = ift.from_random('normal', op.domain)
    inp = f.val
    ret = op(f).val
    assert_(ret[ind] == inp)
    assert_(np.sum(ret) == inp)
