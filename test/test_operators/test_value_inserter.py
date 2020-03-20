# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2019 Max-Planck-Society
#
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
    ift.random.push_sseq_from_seed(seed)
    ind = tuple([int(ift.random.current_rng().integers(0, ss - 1)) for ss in sp.shape])
    op = ift.ValueInserter(sp, ind)
    f = ift.from_random('normal', op.domain)
    ift.random.pop_sseq()
    inp = f.val
    ret = op(f).val
    assert_(ret[ind] == inp)
    assert_(np.sum(ret) == inp)
