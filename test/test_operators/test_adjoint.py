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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import unittest
import nifty4 as ift
import numpy as np
from itertools import product
from test.common import expand
from numpy.testing import assert_allclose


def _check_adjointness(op, dtype=np.float64):
    f1 = ift.Field.from_random("normal", domain=op.domain, dtype=dtype)
    f2 = ift.Field.from_random("normal", domain=op.target, dtype=dtype)
    cap = op.capability
    if ((cap & ift.LinearOperator.TIMES) and
            (cap & ift.LinearOperator.ADJOINT_TIMES)):
        assert_allclose(f1.vdot(op.adjoint_times(f2)),
                        op.times(f1).vdot(f2),
                        rtol=1e-8)
    if ((cap & ift.LinearOperator.INVERSE_TIMES) and
            (cap & ift.LinearOperator.INVERSE_ADJOINT_TIMES)):
        assert_allclose(f1.vdot(op.inverse_times(f2)),
                        op.inverse_adjoint_times(f1).vdot(f2),
                        rtol=1e-8)


_h_RG_spaces = [ift.RGSpace(7, distances=0.2, harmonic=True),
                ift.RGSpace((12, 46), distances=(.2, .3), harmonic=True)]
_h_spaces = _h_RG_spaces + [ift.LMSpace(17)]

_p_RG_spaces = [ift.RGSpace(19, distances=0.7),
                ift.RGSpace((1, 2, 3, 6), distances=(0.2, 0.25, 0.34, .8))]
_p_spaces = _p_RG_spaces + [ift.HPSpace(17), ift.GLSpace(8, 13)]


class Adjointness_Tests(unittest.TestCase):
    @expand(product(_h_spaces, [np.float64, np.complex128]))
    def testPPO(self, sp, dtype):
        op = ift.PowerProjectionOperator(sp)
        _check_adjointness(op, dtype)
        ps = ift.PowerSpace(
            sp, ift.PowerSpace.useful_binbounds(sp, logarithmic=False, nbin=3))
        op = ift.PowerProjectionOperator(sp, ps)
        _check_adjointness(op, dtype)
        ps = ift.PowerSpace(
            sp, ift.PowerSpace.useful_binbounds(sp, logarithmic=True, nbin=3))
        op = ift.PowerProjectionOperator(sp, ps)
        _check_adjointness(op, dtype)

    @expand(product(_h_RG_spaces+_p_RG_spaces,
                    [np.float64, np.complex128]))
    def testFFT(self, sp, dtype):
        op = ift.FFTOperator(sp)
        _check_adjointness(op, dtype)
        op = ift.FFTOperator(sp.get_default_codomain())
        _check_adjointness(op, dtype)

    @expand(product(_h_spaces, [np.float64, np.complex128]))
    def testHarmonic(self, sp, dtype):
        op = ift.HarmonicTransformOperator(sp)
        _check_adjointness(op, dtype)
