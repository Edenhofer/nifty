from __future__ import division
import unittest

import numpy as np
from numpy.testing import assert_equal,\
    assert_allclose,\
    assert_approx_equal

from nifty2go import Field,\
    DiagonalOperator

from test.common import generate_spaces

from itertools import product
from test.common import expand

class DiagonalOperator_Tests(unittest.TestCase):
    spaces = generate_spaces()

    @expand(product(spaces, [True, False]))
    def test_property(self, space, copy):
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(diag)
        if D.domain[0] != space:
            raise TypeError
        if D.unitary != False:
            raise TypeError
        if D.self_adjoint != True:
            raise TypeError

    @expand(product(spaces, [True, False]))
    def test_times_adjoint(self, space, copy):
        rand1 = Field.from_random('normal', domain=space)
        rand2 = Field.from_random('normal', domain=space)
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(diag, copy=copy)
        tt1 = rand1.vdot(D.times(rand2))
        tt2 = rand2.vdot(D.times(rand1))
        assert_approx_equal(tt1, tt2)

    @expand(product(spaces, [True, False]))
    def test_times_inverse(self, space, copy):
        rand1 = Field.from_random('normal', domain=space)
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(diag, copy=copy)
        tt1 = D.times(D.inverse_times(rand1))
        assert_allclose(rand1.val, tt1.val)

    @expand(product(spaces, [True, False]))
    def test_times(self, space, copy):
        rand1 = Field.from_random('normal', domain=space)
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(diag, copy=copy)
        tt = D.times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces, [True, False]))
    def test_adjoint_times(self, space, copy):
        rand1 = Field.from_random('normal', domain=space)
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(diag, copy=copy)
        tt = D.adjoint_times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces, [True, False]))
    def test_inverse_times(self, space, copy):
        rand1 = Field.from_random('normal', domain=space)
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(diag, copy=copy)
        tt = D.inverse_times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces, [True, False]))
    def test_adjoint_inverse_times(self, space, copy):
        rand1 = Field.from_random('normal', domain=space)
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(diag, copy=copy)
        tt = D.adjoint_inverse_times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces, [True, False]))
    def test_diagonal(self, space, copy):
        diag = Field.from_random('normal', domain=space)
        D = DiagonalOperator(diag, copy=copy)
        diag_op = D.diagonal()
        assert_allclose(diag.val, diag_op.val)
